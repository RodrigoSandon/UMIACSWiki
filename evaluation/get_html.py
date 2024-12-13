# scripts/get_html.py
# I do not claim the total ownership of this code, help with AI

"""
This script provides functionality for extracting and cleaning text content from HTML documents. It contains 
three main functions that work together.
"""

import re
from bs4 import BeautifulSoup

NON_BREAKING_ELEMENTS = [
    "a", "abbr", "acronym", "audio", "b", "bdi", "bdo", "big", "button",
    "canvas", "cite", "code", "data", "datalist", "del", "dfn", "em", "embed",
    "i", "iframe", "img", "input", "ins", "kbd", "label", "map", "mark",
    "meter", "noscript", "object", "output", "picture", "progress", "q", "ruby",
    "s", "samp", "script", "select", "slot", "small", "span", "strong", "sub",
    "sup", "svg", "template", "textarea", "time", "u", "tt", "var", "video",
    "wbr",
]

def html_to_text(text, preserve_new_lines=True, strip_tags=["style", "script"]):
    soup = BeautifulSoup(text, "html.parser")
    for element in soup(strip_tags):
        element.extract()
    if preserve_new_lines:
        for element in soup.find_all():
            strings = element.find_all(string=True, recursive=False)
            if strings:
                # if the element is not a non-breaking element and has strings
                if element.name not in NON_BREAKING_ELEMENTS:
                    # add new lines to separate block elements
                    if element.name == "br":
                        element.append("\n")
                    else:
                        element.append("\n\n")
    return soup.get_text(separator="")

def replace_newlines(text):
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def get_html(file):
    with open(file, "r") as f:
        soup = BeautifulSoup(f, "html.parser")
        iwant = soup.find_all("div", {"id": "content"})
        if len(iwant) != 1:
            # fallback: if "div#content" not found, use entire page
            text = html_to_text(str(soup))
        else:
            text = html_to_text(str(iwant[0]))
        text = replace_newlines(text).strip()
        return text
    return "?"
