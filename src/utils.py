import numpy as np

class LoggerWriter:
    def __init__(self, logfct):
        self.logfct = logfct
        self.buf = []

    def write(self, msg):
        if msg.endswith('\n'):
            self.buf.append(msg.rstrip('\n'))
            self.logfct(''.join(self.buf))
            self.buf = []
        else:
            self.buf.append(msg)

    def flush(self):
        pass