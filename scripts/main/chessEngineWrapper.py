import subprocess
import sys


class ChessEngineWrapper(subprocess.Popen):

    def __init__(self, chessEnginePath=''):
        try:
            subprocess.Popen.__init__(self, chessEnginePath, universal_newlines=True,
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE)
        except Exception:
            sys.exit('Path of chess engine is not correct!!!!!!!')

        print (self.uci())

    def sendCommand(self, command):
        self.stdin.write(command + '\n')
        self.stdin.flush()

    def flush(self):
        self.stdout.flush()

    def uci(self):
        self.sendCommand('uci')
        while True:
            line = self.stdout.readline().strip()
            if line == 'uciok':
                return line


    def setposition(self, fenString):
        try:
            self.sendCommand(f"position fen {fenString}")
            # self.isready()
            while True:
                line = self.stdout.readline().strip()
                print(line)
                if 'Fen' in line or "CurrentPlayer" in line:
                    return 

        except ValueError as e:
            print('\nCheck position correctness\n')
            sys.exit(e.message)


    def go(self, moveTime):
        self.sendCommand(f'go movetime {moveTime}')
        while True:
            line = self.stdout.readline().strip()
            print(line)
            if 'bestmove' in line:
                return 
                

    def isready(self):
        self.sendCommand('isready')
        while True:
            line = self.stdout.readline().strip()
            if line == 'readyok':
                return line

    def ucinewgame(self):
        self.sendCommand('ucinewgame')
        self.isready()

    # def printPosition(self):
    #     self.sendCommand("d")
    #     while True:
    #         line = self.stdout.readline().strip()
    #         print(line)
    #         if 'Fen' in line or "CurrentPlayer" in line:
    #             return 