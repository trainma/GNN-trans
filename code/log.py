import logging


class Log():
    def __init__(self, name, file_path, level="DEBUG"):
        self.log = logging.getLogger(name=name)
        self.log.setLevel(level)
        self.file_path = file_path
    def console_handle(self, level="DEBUG"):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(self.get_formatter()[0])
        return console_handler

    def file_handle(self, level="DEBUG"):
        file_handle = logging.FileHandler(self.file_path, mode="a", encoding='utf-8')
        file_handle.setLevel(level)
        file_handle.setFormatter(self.get_formatter()[1])
        return file_handle

    def get_formatter(self):
        console_fmt = logging.Formatter(fmt="%(levelname)s---->%(name)s--->%(asctime)s--->%(message)s")
        file_fmt = logging.Formatter(fmt="%(levelname)s---->%(name)s--->%(asctime)s--->%(message)s")
        return console_fmt, file_fmt

    def get_log(self):
        self.log.addHandler(self.console_handle())
        self.log.addHandler(self.file_handle())

        return self.log


if __name__ == '__main__':
    log = Log('train')
    logger = log.get_log()
    logger.info("print test info ")
    logger.info("finsih!")
