
import sys

from app import MainWindow

if __name__ == "__main__":
    app = MainWindow(sys.argv)
    sys.exit(app.exec_())
