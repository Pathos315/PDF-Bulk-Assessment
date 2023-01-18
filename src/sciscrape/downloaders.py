r"""
Downloads papers en masse
"""
import random
from tempfile import TemporaryFile
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from os import path
from time import sleep
import re
from typing import Optional
from requests import Response
from selectolax.parser import HTMLParser

from sciscrape.webscrapers import client
from sciscrape.log import logger
from sciscrape.change_dir import change_dir
from sciscrape.config import FilePath, config


@dataclass(frozen=True, order=True)
class DownloadReceipt:
    """
    A representation of the receipt describing whether
    or not the download was successful, and,
    if so, where the ensuing file may be found.

    Attributes
    ---------
    downloader : str
        The class name of the downloader
        (e.g. 'BulkPDFDownloader, ImageDownloader')
    success : bool
        If the download was successful or not. Defaults to False.
    filepath : str
        Where the file is located if downloaded. Defaults to 'N/A'.
    """

    downloader: str
    success: bool = False
    filepath: str = "N/A"

    @classmethod
    def from_dict(cls, dict_input: dict):
        return cls(**dict_input)

    def to_dict(self):
        return asdict(self)


@dataclass
class Downloader(ABC):
    """An abstract representation of a scraper that downloads files."""

    url: str
    sleep_val: float = config.sleep_interval
    cls_name: str = field(init=False)
    export_dir: str = config.export_dir

    def __post_init__(self) -> None:
        self.cls_name = self.__class__.__name__

    @abstractmethod
    def obtain(self, search_text: str) -> Optional[DownloadReceipt]:
        """
        For downloaders, `obtain` submits a payload,
        as acquired from the prior search terms,
        to the configured downloader website.
        If the request is successful,
        it isolates the link to download as a link of
        its own, and makes that
        request. Finally, the paper is downloaded to the
        configured export directory,
        while a receipt of this download is
        passed back to the dataframe.
        \n
        The `DownloadReceipt` describes whether
        or not the download was successful, and,
        if so, where the ensuing .pdf may be found.

        Parameters
        ----------
        search_text : str
            Often is the pubid or digital object identifier (DOI)
            of the paper in question. Might
            otherwise be a URL as a string.

        Returns
        -------
        DownloadReceipt : dataclass
            A dataclass that describes the `Downloader` used,
            whether or not the download was successful, and,
            if so, where that file may be found.
        """

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
        with (change_dir(self.export_dir), TemporaryFile() as temp):
            temp.write(contents)
            with open(filename, "wb") as file:
                for line in temp:
                    file.write(line)


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

    link_cleaning_pattern: re.Pattern[str] = re.compile(
        r"(?P<location>location\.href=\')(?P<sep>/+)?"
    )

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
        payload: dict[str, str] = {"request": search_text}
        paper_title: str = f'{config.today}_{search_text.replace("/","")}.pdf'
        response_text: Optional[str] = self.get_response(payload)

        download_link: Optional[str] = (
            None if response_text is None else self.find_download_link(response_text)
        )
        formatted_src: Optional[str] = (
            None if download_link is None else self.format_download_link(download_link)
        )
        logger.debug("download_link=%s", formatted_src)
        return (
            DownloadReceipt(self.cls_name)
            if formatted_src is None
            else self.download_paper(paper_title, formatted_src)
        )

    def download_paper(self, paper_title: str, formatted_src: str) -> DownloadReceipt:
        paper_contents: bytes = client.get(formatted_src, stream=True).content
        self.create_document(paper_title, paper_contents)
        return DownloadReceipt(self.cls_name, True, f"{self.export_dir}/{paper_title}")

    def get_response(self, payload: dict[str, str]) -> Optional[str]:
        response: Response = client.post(
            self.url,
            data=payload,
        )
        sleep(self.sleep_val)
        logger.debug(
            "response=%s, scraper=%s, status_code=%s",
            repr(response),
            repr(self),
            response.status_code,
        )
        return None if response.status_code != 200 else response.text

    def find_download_link(self, search_text: str) -> Optional[str]:
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
        html: HTMLParser = HTMLParser(search_text)
        try:
            download_link: Optional[str] = html.css_first(
                "#buttons button:nth-child(1)"
            ).attributes["onclick"]
            logger.debug("download_link=%s", download_link)
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

    def format_download_link(self, download_link: Optional[str]) -> Optional[str]:
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
        link_match_object = (
            None if download_link is None else self.clean_link_with_regex(download_link)
        )
        return (
            None
            if not link_match_object or not download_link
            else self.adjust_download_link(download_link, link_match_object)
        )

    def adjust_download_link(
        self, download_link: str, link_match_object: re.Match[str]
    ) -> str:
        location_href = link_match_object.group(1)
        seperator = link_match_object.group(2)

        download_link = download_link.replace(location_href, "")

        if seperator == "//":
            download_link = download_link.replace(seperator, "https://", 1)
        else:
            # Replace the `seperator` with `self.url`
            download_link = download_link.replace(seperator, self.url, 1)
        return download_link

    def clean_link_with_regex(self, download_link: str) -> Optional[re.Match[str]]:
        return self.link_cleaning_pattern.match(download_link)


@dataclass
class ImagesDownloader(Downloader):
    """
    The ImagesDownloader class takes the provided
    string from a prior list.
    Using that string value, it posts it to the selected website.
    Then, it downloads the image
    file that appears as a result of that query.
    """

    def obtain(self, search_text: str) -> Optional[DownloadReceipt]:
        sleep(self.sleep_val)
        search_ext = search_text.split(".")[-1]
        response: Response = client.get(search_text, stream=True, allow_redirects=True)
        logger.debug(
            "response=%s, scraper=%s, status_code=%s",
            repr(response),
            repr(self),
            response.status_code,
        )

        return (
            DownloadReceipt(self.cls_name)
            if response.status_code != 200
            else self.download_image(search_ext, response)
        )

    def download_image(self, search_ext: str, response: Response) -> DownloadReceipt:
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
        filename: str = self.format_filename(response.headers.get("Etag"), search_ext)
        self.create_document(filename, response.content)
        fullpath: str = str(path.abspath(filename))
        return DownloadReceipt(self.cls_name, True, fullpath)

    def format_filename(self, etag: Optional[str], ext: str) -> str:
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

        file_id: int = random.randint(1, 255)
        if etag:
            etag = etag.strip('"')
            filename: str = f"{config.today}_{etag}_{file_id}.{ext}"
        else:
            filename = f"{config.today}__NaN__{file_id}.{ext}"
        logger.debug("filename=%s", filename)
        return filename
