# OCR Scraper for The Last War

## Usage

```shell
>>> python3 last_war.py --help
Usage: last_war.py [OPTIONS] [INPUT]...

Options:
  -v, --verbose         Enable verbose output.
  --delete              Delete the input files.
  --frame-rate INTEGER  Frame rate for video processing.
  --csv-out TEXT        Output CSV file name.
  --guild TEXT          Guild name to filter the results.
  --similarity FLOAT    Similarity threshold for text matching.
  --help                Show this message and exit.
```

## Installation

- Install requirements
```shell
pip install -r requirements.txt
```

- Install Tesseract
```shell
sudo apt install tesseract-ocr
```

- Update `DEFAULTS`
    - `pytesseract` cmd path
        - `which tesseract` on UNIX
        - `where tesseract` on Windows (I think)
