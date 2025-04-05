import csv
import os
from difflib import SequenceMatcher

import click
import cv2
import pytesseract


# MARK: Defaults
# Valid file types
VALID_IMG = (".jpg", ".png")
VALID_VIDEO = (".mp4", ".avi")
VALID_FILES = (*VALID_IMG, *VALID_VIDEO)

# Default CSV file name
DEFAULT_TESSERACT = (
    "/usr/bin/tesseract"  # r'C:\Program Files\Tesseract-OCR\tesseract.exe'
)
DEFAULT_GUILD = "[Gvo] Good Vibes Only"
DEFAULT_CSV = "versus_scores.csv"
DEFAULT_FRAME_RATE = 1
DEFAULT_SIMILARITY = 0.5

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT


# MARK: Cropping
def identify_crop_coords(frame: cv2.Mat) -> tuple[int, int, int, int]:
    """Identify the crop coordinates for the left and right of the commander and points they scored"""

    # Interactive ROI selection, returns list[(x, y, w, h)]
    rois = cv2.selectROIs(
        "Identify crop regions",
        frame,
        fromCenter=False,
        showCrosshair=True,
        printNotice=True,
    )
    cv2.destroyWindow("Identify crop regions")

    if len(rois) != 2:
        click.secho(
            "\nPlease identify the commander and point regions only.\n",
            bold=True,
            err=True,
        )
        return identify_crop_coords(frame)

    # return x1, x1 + w1, x2, x2 + w2
    return (
        int(rois[0][0]),
        int(rois[0][0] + rois[0][2]),
        int(rois[1][0]),
        int(rois[1][0] + rois[1][2]),
    )


# MARK: Stitching
def stitch_frames(frames: list[cv2.Mat], frame_rate: int) -> list[cv2.Mat]:
    """
    Attempt frame stitching to reduce the number of frames. Returns the stitched images
    """
    # Create a stitcher object
    stitcher = cv2.Stitcher.create(
        mode=cv2.Stitcher_PANORAMA
    )  # TODO: Use cv2.Stitcher_SCANS since no corrections?

    # Stitch the frames together
    status, stitched_image = stitcher.stitch(frames)

    if status == cv2.Stitcher_OK:
        return [stitched_image]

    click.echo("Stitching failed", err=True)
    return frames
    #     curr_frame: int = 1
    # while cap.isOpened():
    #     ret, frame = cap.read()

    #     if not ret:
    #         break

    #     if not curr_frame % 1:
    #         is_good, new_stitch = stitcher.stitch([stitched_image, frame])
    #         if is_good != cv2.STITCHER_OK:
    #             cv2.imshow("next image", frame)
    #             cv2.imshow("Stitching failed", stitched_image)
    #             cv2.waitKey(10_000)

    #             raise RuntimeError("Stitching failed at frame: ", curr_frame)

    #         stitched_image = new_stitch
    #         cv2.imwrite(f"stitch{curr_frame:0>3}.png", stitched_image)

    #     curr_frame += 1

    # cap.release()

    # return stitched_image


# MARK: Text Extraction
def extract_text_from_frames(
    commander_frame: cv2.Mat,
    points_frame: cv2.Mat,
) -> tuple[str, str]:
    """Convert frames to text using OCR."""
    commander_text = pytesseract.image_to_string(commander_frame)
    points_text = pytesseract.image_to_string(points_frame)
    # print(f"Extracted text: {text}")  # Debugging line to check OCR output
    return (commander_text, points_text)


# MARK: Text Parsing
def text_parser(
    texts: list[tuple[str, str]],
    guild: str = DEFAULT_GUILD,
    similarity: float = DEFAULT_SIMILARITY,
) -> dict[str, dict[int, int]]:
    def is_similar(a: str, b: str, threshold: float = similarity) -> bool:
        return SequenceMatcher(None, a, b).ratio() >= threshold

    # NOTE: The problem with the ocr is that I've found it works best with close cropping.
    # This means that the commanders and points are identified separately.
    # This leads to variation in the identifiactions across frames.
    # To solve this, I believe that the best solution is to:
    # 1. store the commanders in a defaultdict (undefined commanders will return 0)
    # 2. For a set of commanders and points in a frame pair, assign all points to all commanders
    # 3. After struct creation, start collapsing the points
    #     3.1 Assign most common points to each commander
    #     3.2 Remove the assigned points from all other commanders
    #     3.3 Repeat until all points are assigned or no collapse is possible
    # 4. Return the results, and a dict of commanders unable to assign points to
    results: dict[str, dict[int, int]] = {}
    for tc, tp in texts:
        # Remove any unwanted characters and split by whitespace
        tc = [
            val
            for val in tc.strip().splitlines()
            if val != "" and not is_similar(val, guild)
        ]
        tp = [
            int(val)
            for val in tp.strip().replace(",", "").replace(".", "").splitlines()
            if val != "" and val.isdigit()
        ]

        for commander in tc:
            # If the commander is not already in `results`, add it
            if commander not in results:
                results[commander] = {}

            for points in tp:
                # If the points are not already in `results`, add it
                if points not in results[commander]:
                    results[commander][points] = 0

                # Increment the points for the commander
                results[commander][points] += 1

    return results


# MARK: MAIN
@click.command()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output.")
@click.option("--delete", is_flag=True, help="Delete the input files.")
@click.option(
    "--frame-rate",
    type=int,
    default=DEFAULT_FRAME_RATE,
    help="Frame rate for video processing.",
)
@click.option("--csv-out", default=DEFAULT_CSV, help="Output CSV file name.")
@click.option(
    "--guild",
    default=DEFAULT_GUILD,
    help="Guild name to filter the results.",
)
@click.option(
    "--similarity",
    type=float,
    default=DEFAULT_SIMILARITY,
    help="Similarity threshold for text matching.",
)
@click.argument(
    "input", nargs=-1
)  # TODO: Use `click.Path(exists=True)` to check if the file exists?
def main(verbose, delete, frame_rate, csv_out, guild, similarity, input) -> int:
    # Parse input, returning a sequence of cropped MatLike objects
    if verbose:
        click.echo("========= Start ========")
        click.echo("0.\tParsing inputs")

    # # No input : search current directory for an mp4 file
    if not input:
        # Search current directory for an mp4 file
        input = [
            file for file in os.listdir(".") if any(map(file.endswith, VALID_FILES))
        ]
        # Return if no mp4 files are found
        if not input:
            click.echo("No mp4 files found in the current directory.", err=True)
            return 1

    # # 1+ inputs: Ensure they exist
    for file in input:
        if not os.path.isfile(file):
            click.echo(f"File {file} does not exist.", err=True)
            return 1

    if verbose:
        click.echo(f"0.0\tInputs identified:\n\t\t{'\n\t\t'.join(input)}")

    # Open inputs
    frames = []
    for file in input:
        # Open image and append to `frames`
        if any(map(file.endswith, VALID_IMG)):
            frames.append(cv2.imread(file))

        # Open video and append every `frame_rate`-th frame to `frames`
        elif any(map(file.endswith, VALID_VIDEO)):
            cap = cv2.VideoCapture(file)
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # TODO: Stitch input videos (reducing # of frames)
                # TODO: Find ideal `frame_rate` based on frame diff?
                if frame_count % frame_rate == 0:
                    frames.append(frame)

                frame_count += 1

            cap.release()

        else:
            click.echo(
                f"File {file} is not recognized as a valid image or video file.",
                err=True,
            )
            click.secho("This may be a bug!", bold=True, err=True)

    comm_left, comm_right, points_left, points_right = identify_crop_coords(frames[0])

    if verbose:
        click.echo(
            ("0.1\tCropping inputs to:")
            + (f"\n\t\tcomm.\t{comm_left, comm_right}")
            + (f"\n\t\tpoints\t{points_left, points_right}")
        )
    commanders = []
    points = []
    for frame in frames:
        # Crop the frame to the identified regions
        comm_col = frame[:, comm_left:comm_right]
        points_col = frame[:, points_left:points_right]

        # Convert the cropped images to grayscale
        comm_col = cv2.cvtColor(comm_col, cv2.COLOR_BGR2GRAY)
        points_col = cv2.cvtColor(points_col, cv2.COLOR_BGR2GRAY)

        # Append the cropped images to the respective lists
        commanders.append(comm_col)
        points.append(points_col)

    assert len(commanders) == len(points), "Mismatch between commanders and points"
    del frames  # Free up memory (Though **should be** garbage collected)

    # Extract text from input, returning a list of tuples
    texts = []
    total_frames = len(commanders)

    if verbose:
        click.echo("1.\tIdentifying text")
        click.echo(f"1.0\tProcessing: {0:>5.1%}", nl=False)

    for i, (commander, point) in enumerate(zip(commanders, points)):
        if verbose:
            click.echo("\b" * 5 + f"{(i + 1) / total_frames:>5.1%}", nl=False)
        texts.append(extract_text_from_frames(commander, point))

    if verbose:
        click.echo("\b" * 6 + f" {1:>5.0%}")

    assert len(texts) == len(commanders), "Mismatch between texts and frames"
    del commanders, points  # Free up memory (Though **should be** garbage collected)

    if verbose:
        click.echo("1.1\tParsing text")

    results = text_parser(texts, guild, similarity)

    # Store text in a CSV file
    if verbose:
        click.echo("2.\tStoring text")

    fields = ["Commander", "Points"]
    data = []
    for commander, points in results.items():
        point = max(points, key=points.get)
        data.append((commander, point))

    with open(csv_out, "w") as f:
        writer = csv.writer(f)

        writer.writerow(fields)
        writer.writerows(data)

    # If delete is set, delete the input files
    if delete:
        if verbose:
            click.echo("3.\tCleaning up")

        for file in input:
            if verbose:
                click.echo("3.0\tDeleting:")
            try:
                os.remove(file)
                if verbose:
                    click.echo(f"\t\t{file}")
            except OSError as e:
                click.echo(f"3.0\tError deleting {file}: {e}")
                return 1

    # Notify user of completion
    if verbose:
        click.echo("======= Finished =======\a\n")

    return 0


# MARK: Entry point
if __name__ == "__main__":
    main()
