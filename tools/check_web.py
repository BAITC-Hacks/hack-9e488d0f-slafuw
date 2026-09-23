"""Check the served HTML's inline JavaScript syntax using Node, without executing it."""

from html.parser import HTMLParser
import subprocess

from eventmatch.catalog import ROOT


class Scripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_script = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.in_script = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False

    def handle_data(self, data):
        if self.in_script:
            self.parts.append(data)


def main():
    parser = Scripts()
    parser.feed((ROOT / "web/index.html").read_text(encoding="utf-8"))
    subprocess.run(["node", "--check"], input="\n".join(parser.parts), text=True,
                   encoding="utf-8", check=True)
    print("OK: served UI JavaScript syntax")


if __name__ == "__main__":
    main()
