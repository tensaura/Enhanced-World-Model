from enum import StrEnum
from sys import stdout
from typing import TextIO


class Style(StrEnum):
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    BLUE = "\033[0;34m"
    MAGENTA = "\033[0;35m"
    CYAN = "\033[0;36m"
    WHITE = "\033[0;37m"
    RESET = "\033[0m"

    BLACK_BG = "\033[40m"
    RED_BG = "\033[41m"
    GREEN_BG = "\033[42m"
    YELLOW_BG = "\033[43m"
    BLUE_BG = "\033[44m"
    MAGENTA_BG = "\033[45m"
    CYAN_BG = "\033[46m"
    WHITE_BG = "\033[47m"

    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    INVERT = "\033[7m"


class Logger:
    def __init__(self, fd: TextIO = stdout) -> None:
        self.fd: TextIO = fd

    def log(self, *messages: str, style: str = Style.GREEN, sep: str = "\n", end: str = "\n") -> None:
        print(f"{style}{sep.join(messages)}{Style.RESET}", file=self.fd, end=end)

    def warn(self, *messages: str, sep: str = "\n") -> None:
        print(f"{Style.YELLOW}{Style.BOLD}Warning: {sep.join(messages)}{Style.RESET}", file=self.fd)

    def error(self, *messages: str, sep: str = "\n") -> None:
        print(f"{Style.RED}{Style.INVERT}Error: {sep.join(messages)}{Style.RESET}", file=self.fd)

    def input(self, *messages: str, style: str = Style.GREEN, sep: str = "\n", end: str = "") -> str:
        self.log(f"{sep.join(messages)}", style=style, end=end)
        return input()

    def dict_log(
        self, data: dict[str, str], key_style: str = Style.CYAN, value_style: str = Style.YELLOW, sep: str = "\n"
    ) -> None:
        print(
            *(f"{key_style}{key}: {value_style}{value}{Style.RESET}" for key, value in data.items()),
            file=self.fd,
            sep=sep,
        )
