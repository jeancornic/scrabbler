
## Scrabbler

The idea here is to solve a scrabble.

Two main steps:
* Detect and recognize letters in a image (grid and "hand")
* Find the best word

For now, it needs a screenshot of android EA Scrabble © game.

### Install & run

```bash
pip install -r requirements.txt
apt-get install python-opencv
python find_letters.py path/to/my/image
```

### Improvments

* Make it work on any scrabble grid : photo ??
