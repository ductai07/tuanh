import dcase_util

class Logger:
    def __init__(self, name="Sound Analysis"):
        self.log = dcase_util.utils.FancyLogger()
        self.log.title(name)
    
    def section(self, title):
        self.log.section_header(title)
    
    def info(self, message):
        self.log.info(message)
    
    def warning(self, message):
        self.log.warning(message)
    
    def error(self, message):
        self.log.error(message)
    
    def debug(self, message):
        self.log.debug(message)