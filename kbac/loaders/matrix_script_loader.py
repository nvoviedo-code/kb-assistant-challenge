import regex
import joblib

from tqdm import tqdm
from itertools import groupby
from more_itertools import flatten
from pypdf import PdfReader, PageObject

from typing import Literal
from pydantic import BaseModel, NonNegativeInt, PositiveInt, StrictStr, Field


class LineItem(BaseModel):
    page_number: NonNegativeInt
    text_type: Literal[
        "location",
        "description",
        "character",
        "dialog",
    ]
    text: StrictStr = Field(min_length=1)
    margin: PositiveInt


class Document(BaseModel):
    text: StrictStr
    metadata: dict


class MatrixScriptLoader:
    def __init__(
        self,
        source_path: str = "/resources/movie-scripts/the-matrix-1999.pdf",
        ignoread_tags: list[str] = [
            "FADE IN:",
            "CONTINUED",
            "OMITTED",
            "THE MATRIX - Rev.",
            "FADE OUT.",
            "THE END",
            "(MORE)",
            "FADE TO BLACK.",
        ],
        location_margins: set[int] = {8, 9},
        description_margins: set[int] = {8, 9},
        dialog_margins: set[int] = {21, 30},
        character_margins: set[int] = {32, 38, 39},
        start_page: int | None = 1,
        end_page: int | None = None,
        show_no_matched_texts: bool = False,
    ):
        super().__init__()

        self.source_path = source_path
        self.ignoread_tags = ignoread_tags
        self.location_margins = location_margins
        self.description_margins = description_margins
        self.dialog_margins = dialog_margins
        self.character_margins = character_margins

        self.start_page = start_page
        self.end_page = end_page
        self.show_no_matched_texts = show_no_matched_texts

    def _get_location_text(
        self,
        line_text: str,
    ) -> tuple[str | None, int | None]:
        if not line_text.isupper():
            return None, None

        match_ = regex.match(
            r"([AB]?\d+\s{2,})(.*?)(\s{2,}[AB]?\d+$)", line_text
        )

        if match_ is None:
            return None, None

        groups = match_.groups()

        margin = len(groups[0])
        if margin not in self.location_margins:
            return None, None

        return groups[1].strip(), margin

    def _get_character_text(
        self,
        line_text: str,
    ) -> tuple[str | None, int | None]:
        if not line_text.isupper():
            return None, None

        if not line_text.endswith(""):
            return None, None

        match_ = regex.match(r"(\s{2,})(.*)", line_text)
        if match_ is None:
            return None, None

        groups = match_.groups()
        margin = len(groups[0])

        if margin not in self.character_margins:
            return None, None

        character_text = groups[1]
        character_text = regex.sub(r"\(.*\)", "", character_text)
        character_text = character_text.strip()

        return character_text, margin

    def _get_non_upper_text(
        self,
        line_text: str,
        margins: set[int],
    ) -> tuple[str, int] | tuple[None, None]:
        match_ = regex.match(r"(\s{2,})(.*)", line_text)
        if match_ is None:
            return None, None

        groups = match_.groups()
        margin = len(groups[0])
        if margin not in margins:
            return None, None

        return groups[1].strip(), margin

    def _get_description_text(
        self,
        line_text: str,
    ) -> tuple[str, int] | tuple[None, None]:
        return self._get_non_upper_text(
            line_text,
            margins=self.description_margins,
        )

    def _get_dialog_text(
        self,
        line_text: str,
    ) -> tuple[str, int] | tuple[None, None]:
        return self._get_non_upper_text(
            line_text,
            margins=self.dialog_margins,
        )

    def _parse_page_line(
        self,
        line_text: str,
        page_number: int,
    ) -> LineItem | None:
        if any(it in line_text for it in self.ignoread_tags):
            return

        location_text, location_margin = self._get_location_text(
            line_text=line_text
        )

        if location_text is not None and location_margin is not None:
            return LineItem(
                page_number=page_number,
                text_type="location",
                text=location_text,
                margin=location_margin,
            )

        character_text, character_margin = self._get_character_text(
            line_text=line_text
        )

        if character_text is not None and character_margin is not None:
            return LineItem(
                page_number=page_number,
                text_type="character",
                text=character_text,
                margin=character_margin,
            )

        description_text, description_margin = self._get_description_text(
            line_text=line_text
        )

        if description_text is not None and description_margin is not None:
            return LineItem(
                page_number=page_number,
                text_type="description",
                text=description_text,
                margin=description_margin,
            )

        dialog_text, dialog_margin = self._get_dialog_text(line_text=line_text)
        if dialog_text is not None and dialog_margin is not None:
            return LineItem(
                page_number=page_number,
                text_type="dialog",
                text=dialog_text,
                margin=dialog_margin,
            )

        if self.show_no_matched_texts:
            no_matched_text = line_text.strip()
            if len(no_matched_text):
                show_item = {
                    "text": line_text,
                    "page_number": page_number,
                }

                print(f"no matched text: {show_item}")

    def parse_page(self, page: PageObject) -> list[LineItem]:
        assert page.page_number is not None
        if self.start_page is not None:
            if page.page_number < self.start_page:
                return []

        if self.end_page is not None:
            if page.page_number > self.end_page:
                return []

        page_number = page.page_number
        page_text = page.extract_text(extraction_mode="layout")

        line_items = (
            self._parse_page_line(
                line_text=line_text,
                page_number=page_number,
            )
            for line_text in page_text.split("\n")
        )

        return [li for li in line_items if li is not None]

    def _agg_line_groups(self, line_group: list[LineItem]) -> LineItem:
        if len(line_group) == 1:
            return line_group[0]

        text_type = line_group[0].text_type
        assert text_type not in {
            "location",
            "character",
        }, f"line_group: {line_group}"

        return LineItem(
            page_number=line_group[0].page_number,
            text_type=text_type,
            text=" ".join(li.text for li in line_group),
            margin=line_group[0].margin,
        )

    def load(self) -> list[Document]:
        reader = PdfReader(self.source_path)
        parsed_pages = map(
            self.parse_page,
            tqdm(
                reader.pages,
                total=reader.get_num_pages(),
            ),
        )

        line_items = list(flatten(parsed_pages))
        line_groups = groupby(line_items, key=lambda x: (x.text_type, x.margin))
        line_items = [
            self._agg_line_groups(line_group=list(lg[1])) for lg in line_groups
        ]

        location = None
        scene_description_id = None
        character = None

        documents = []
        for idx, li in enumerate(line_items, start=1):
            text = li.text
            text_type = li.text_type
            if text_type == "location":
                location = text
                continue

            if text_type == "character":
                character = text
                continue

            if text_type == "description":
                scene_description_id = joblib.hash(text)
                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "text_type": "scene_description",
                            "scene_description_id": scene_description_id,
                            "character": None,
                            "location": location,
                            "page_number": li.page_number,
                            "line_number": idx,
                        },
                    )
                )

                continue

            if text_type == "dialog":
                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "text_type": text_type,
                            "character": character,
                            "location": location,
                            "scene_description_id": scene_description_id,
                            "page_number": li.page_number,
                            "line_number": idx,
                        },
                    )
                )

        return documents
