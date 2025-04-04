import os

import click

# import cv2
# import numpy as np
# import pytesseract


@click.command()
@click.option("--verbose", is_flag=True, help="Enable verbose output.")
@click.option("--delete", is_flag=True, help="Delete the input files.")
@click.argument("input", nargs=-1, help="Input video or image files.")
def main(verbose, delete, input) -> int:
    # Parse input, returning a sequence of cropped MatLike objects
    if verbose:
        click.echo("========= Start ========")
        click.echo("0.\tParsing inputs")

    # # No input : search current directory for an mp4 file
    if not input:
        # Search current directory for an mp4 file
        input = [file for file in os.listdir(".") if file.endswith(".mp4")]
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
        click.echo(f"0.0\tInputs identified: {input}")

    # TODO: Identify crop regions (left/right for both commander and points)
    # TODO: Crop all inputs

    # TODO: Stitch input videos (to reduce the number of frames)
    # Find ideal frame rate based on frame difference?

    # Extract text from input, returning a list of tuples
    if verbose:
        click.echo("1.\tIdentifying text")

    # TODO: Implement text extraction using Tesseract

    # Store text in a CSV file
    if verbose:
        click.echo("2.\tStoring text")

    # TODO: Implement CSV storage

    # If delete is set, delete the input files
    if delete:
        if verbose:
            click.echo("3.\tCleaning up")

        for file in input:
            try:
                os.remove(file)
            #     if verbose:
            #         click.echo(f"Deleted {file}")
            except OSError as e:
                click.echo(f"Error deleting {file}: {e}")
                return 1

    # Notify user of completion
    if verbose:
        click.echo("======= Finished =======\a\n")

    return 0


if __name__ == "__main__":
    main()
