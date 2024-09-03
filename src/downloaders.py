"""
Downloads papers en masse
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryFile
from time import sleep
from typing import TYPE_CHECKING

from selectolax.parser import HTMLParser

from src.change_dir import change_dir
from src.config import FilePath, config
from src.log import log_debug, logger
from src.scraperesults import DownloadReceipt
from src.webscrapers import WebScraper, client

if TYPE_CHECKING:
    from requests import Response


LINK_CLEANING_PATTERN = re.compile(
    r"(?P<location>location\.href=\')(?P<sep>/+)?"
)


@dataclass
class Downloader(WebScraper):
    """An abstract representation of a scraper that downloads files."""

    sleep_val: float = 1.0
    cls_name: str = field(init=False)
    export_dir: FilePath = Path(config.export_dir)

    def __post_init__(self) -> None:
        self.cls_name = type(self).__name__

    def create_document(
        self,
        filename: FilePath,
        contents: bytes,
    ) -> None:
        """
        `create_document` goes to, and downloads, the isolated download link.
        It is sent as bytes to a temporary text file.
        The temporary text file is then used as a basis to generate a new pdf.
        Afterwards, the temporary text file is deleted
        in preparation for the next pdf.

        Parameters
        ----------
        search_text : str
            The initial term to be rendered in the filename.
        link : str
            The link to be followed.

        Returns
        -------
            A .pdf or .png file, depending on the `Downloader` in use.
        """
        with change_dir(self.export_dir), TemporaryFile() as temp:
            temp.write(contents)
            with open(filename, "wb") as file:
                file.writelines(temp)

    def format_request(self):
        raise NotImplementedError("Not applicable to this class.")

    def process_response(self):
        raise NotImplementedError("Not applicable to this class.")


@dataclass
class BulkPDFScraper(Downloader):
    """
    The BulkPDFScrape class takes the provided
    string from a prior list.
    Using that string value, it posts it to the selected website.
    Then, it downloads the ensuing .pdf
    file that appears as a result of that query.

    Attributes
    ----------
    link_cleaning_pattern : Pattern[str]
        the regex pattern that cleans the download link
    """

    link_cleaning_pattern: re.Pattern[str] = LINK_CLEANING_PATTERN

    def obtain(self, search_text: str) -> DownloadReceipt:
        """
        `obtain` submits a payload, as acquired from
        the prior search terms, to the specified downloader website.
        If the request is successful, it isolates the link
        to download as a link of its own, and makes that request.
        Finally, the paper is downloaded to the specified export directory,
        while a receipt of this download is passed back to the dataframe.
        The `DownloadReceipt` describes whether or not
        the download was successful, and,
        if so, where the ensuing .pdf may be found.

        Parameters
        ----------
        search_text : str
            the pubid or digital object identifier
            (DOI) of the paper in question.

        Returns
        -------
        DownloadReceipt : dataclass
            A dataclass describing whether or not
            the download was successful, and,
            if so, where the ensuing .pdf may be found.
        """
        payload = {"request": search_text}
        paper_title = Path(f"{config.today}_{search_text.replace('/','')}.pdf")
        response_text = self.make_request(
            url=self.url,
            method="POST",
            payload=payload,
        )

        download_link: str | None = self.find_download_link(response_text)
        formatted_src: str | None = self.format_download_link(download_link)
        return (
            self.download_paper(paper_title, formatted_src)
            if formatted_src
            else DownloadReceipt(self.cls_name)
        )

    def download_paper(
        self, paper_title: FilePath, formatted_src: str
    ) -> DownloadReceipt:
        paper_response = self.make_request(url=formatted_src, stream=True)
        if not (paper_contents := paper_response.content):  # type: ignore
            return DownloadReceipt(self.cls_name, False)
        self.create_document(paper_title, paper_contents)
        return DownloadReceipt(
            self.cls_name, True, f"{self.export_dir}/{paper_title}"
        )

    @log_debug
    def find_download_link(self, search_text: str | None) -> str | None:
        """
        create_querystring, within `BulkPDFScraper`,
        returns a link that will download the paper in question.

        Parameters
        ---------
        resp_text : text
            A prior Response's text, as HTML, to be parsed.

        Return
        -----
        str
            A download link, which will download a link to the paper.
        """
        if not isinstance(search_text, str):
            return None
        html = HTMLParser(search_text)
        try:
            download_link: str | None = html.css_first(
                "#buttons button:nth-child(1)"
            ).attributes["onclick"]
            return download_link
        except (AttributeError, ValueError) as e:
            logger.error(
                'No "onclick" attribute found within downloader=%s DOM attributes.\
                        The following error occurred in BulkPDFScraper while accessing the paper\'s link: error=%s.\
                        Proceeding to next in sequence.',
                self.url,
                e,
            )
            return None

    def format_download_link(self, download_link: str | None) -> str | None:
        """
        format_download_link first cleans the download_link
        according to the provided regular expression pattern.

        It then either formats it as a corrected URL string if it is valid, or returns None
        if it fails.

        Parameters
        ---------
        download_link : str
            An unformatted download link, which will
            ultimately download a link to the paper.
        Return
        -----
        str
            A link to the requested academic paper.
        """
        if not isinstance(download_link, str):
            return None
        link_match_object = self.clean_link_with_regex(download_link)
        return (
            self.adjust_download_link(download_link, link_match_object)
            if link_match_object
            else None
        )

    def adjust_download_link(
        self, download_link: str, link_match_object: re.Match[str]
    ) -> str:
        location_href = link_match_object.group(1)
        seperator = link_match_object.group(2)

        download_link = download_link.replace(location_href, "")
        download_link = (
            download_link.replace(seperator, "https://", 1)
            if seperator == "//"
            else download_link.replace(seperator, self.url, 1)
        )
        return download_link

    def clean_link_with_regex(
        self, download_link: str | None
    ) -> re.Match[str] | None:
        return (
            self.link_cleaning_pattern.match(download_link)
            if download_link
            else None
        )


@dataclass
class ImagesDownloader(Downloader):
    """
    The ImagesDownloader class takes the provided
    string from a prior list.
    Using that string value, it posts it to the selected website.
    Then, it downloads the image
    file that appears as a result of that query.
    """

    @log_debug
    def obtain(self, search_text: str) -> DownloadReceipt | None:
        """
        Queries the downloader website with the given search text,
        and attempts to download the image associated with the search text.

        Parameters:
        search_text (str): The search text to query the downloader website with.

        Returns:
        DownloadReceipt: A receipt indicating whether the image was successfully
        downloaded and the path to the downloaded image.
        """
        sleep(self.sleep_val)
        search_ext = search_text.split(".")[-1]
        response = client.get(search_text, stream=True, allow_redirects=True)

        return (
            self.download_image(search_ext, response)
            if response
            else DownloadReceipt(self.cls_name)
        )

    def download_image(
        self, search_ext: str, response: Response
    ) -> DownloadReceipt:
        """
        Downloads an image from a given HTTP response and stores it on the local file system.

        Parameters
        ---------
        search_ext (str):
            The file extension of the image to be downloaded.
        response (Response):
            The HTTP response object containing the image to be downloaded.

        Returns
        -------
        DownloadReceipt:
            A receipt indicating whether the image was successfully downloaded and the path to the downloaded image.
        """
        filename: FilePath = self.format_filename(
            response.headers.get("Etag"), search_ext
        )
        self.create_document(filename, response.content)
        fullpath = (filename.resolve()).name
        return DownloadReceipt(self.cls_name, True, fullpath)

    def format_filename(self, etag: str | None, ext: str) -> Path:
        """
        format_filename, within `ImageDownloader`, generates a filename
        for the image to be downloaded.

        Parameters
        ---------
        resp_text : text
            A prior Response's text, as HTML, to be parsed.

        Return
        -----
        str
            A filename to which the image will be downloaded.
        """

        file_id = random.randint(1, 255)
        etag = (etag or "_NaN_").strip('"')
        filename = Path(f"{config.today}_{etag}_{file_id}.{ext}")
        logger.debug("filename=%s", filename)
        return filename
