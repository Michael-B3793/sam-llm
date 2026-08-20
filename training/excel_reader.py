import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd


NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkg": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def column_index(cell_ref):
    letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter) - 64
    return value - 1


def parse_scalar(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
        if number.is_integer():
            return int(number)
        return number
    except ValueError:
        return text


def read_shared_strings(workbook):
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return []
    root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    strings = []
    for item in root.findall("main:si", NS):
        parts = [node.text or "" for node in item.findall(".//main:t", NS)]
        strings.append("".join(parts))
    return strings


def first_sheet_path(workbook):
    workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
    rels_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
    sheets = workbook_root.findall("main:sheets/main:sheet", NS)
    if not sheets:
        return "xl/worksheets/sheet1.xml"
    relation_id = sheets[0].attrib[f"{{{NS['rel']}}}id"]
    targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels_root.findall("pkg:Relationship", NS)
    }
    target = targets.get(relation_id, "worksheets/sheet1.xml")
    if target.startswith("/"):
        return target.lstrip("/")
    return str(Path("xl") / target)


def cell_value(cell, shared_strings):
    cell_type = cell.attrib.get("t")
    value_node = cell.find("main:v", NS)
    if cell_type == "inlineStr":
        text_node = cell.find(".//main:t", NS)
        return text_node.text if text_node is not None else None
    raw = value_node.text if value_node is not None else None
    if cell_type == "s" and raw is not None:
        return shared_strings[int(raw)]
    return parse_scalar(raw)


def read_xlsx(path):
    path = Path(path)
    with zipfile.ZipFile(path) as workbook:
        shared_strings = read_shared_strings(workbook)
        sheet_path = first_sheet_path(workbook)
        root = ET.fromstring(workbook.read(sheet_path))
        rows = []
        for row in root.findall(".//main:sheetData/main:row", NS):
            values = {}
            for cell in row.findall("main:c", NS):
                ref = cell.attrib.get("r", "")
                values[column_index(ref)] = cell_value(cell, shared_strings)
            if values:
                rows.append(values)
    if not rows:
        return pd.DataFrame()
    width = max(max(row) for row in rows) + 1
    matrix = [[row.get(i) for i in range(width)] for row in rows]
    header = [str(value).strip() if value is not None else f"column_{i}" for i, value in enumerate(matrix[0])]
    return pd.DataFrame(matrix[1:], columns=header)
