import os
import sys

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QComboBox, \
    QHBoxLayout, QSizePolicy, QMessageBox, QFrame, QGroupBox, QGridLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from data_handling import scrape_data, organize_data_files, get_prediction
from data_handling import load_data, get_pick_rate, get_win_rate

EXECUTABLE_DIRECTORY = sys._MEIPASS


class MainWindow(QApplication):
    PHOTO_HEIGHT = 150
    MAX_BOX_HEIGHT = 20
    MAX_BOX_WIDTH = 150
    BUTTON_STYLESHEET = "background-color: {}; color: white; font-size: 16px; padding: 10px; border-radius: 5px;"
    BUTTON_FIXED_POLICY = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    AGENT_OPTIONS = sorted(
        ['Gekko', 'Deadlock', 'Brimstone', 'Phoenix', 'Sage', 'Sova', 'Viper', 'Cypher', 'Reyna',
         'Killjoy', 'Breach', 'Omen', 'Jett', 'Raze', 'Skye', 'Yoru', 'Astra', 'KAY/O', 'Chamber',
         'Neon', 'Fade', 'Harbor'])

    RANK_CATEGORIES = sorted(
        ["Iron", "Bronze", "Silver", "Gold", "Platinum", "Diamond", "Ascendant", "Immortal", "Radiant"])

    RANKED_MAPS = sorted(["Split", "Ascent", "Haven", "Bind", "Fracture", "Pearl", "Lotus"])

    def __init__(self, argv):
        super().__init__(argv)
        self.setStyle("Fusion")
        self.setWindowIcon(QIcon(os.path.join(EXECUTABLE_DIRECTORY, "..", "..", "application_icon.png")))
        self.main_window = QMainWindow()
        self.main_window.setWindowTitle("Valorant Match Results Predictor")
        self.main_window.setStyleSheet("background-color: #f0f0f0;")
        self.data = {}

        self.central_widget = QWidget()
        self.main_window.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setSpacing(20)

        self.setup_rank_and_map_widgets()
        self.layout.addLayout(self.rank_map_layout)

        central_layout = QHBoxLayout()
        self.layout.addLayout(central_layout)

        left_layout = QVBoxLayout()
        central_layout.addLayout(left_layout)

        right_layout = QVBoxLayout()
        central_layout.addLayout(right_layout)

        self.initialize_teams_widgets(left_layout, right_layout)

        self.setup_graph_widgets()
        self.layout.addLayout(self.graph_layout)

        self.setup_button_widgets()
        self.layout.addLayout(self.button_layout)

        self.main_window.showMaximized()

    # Initializes the widgets for each of the two teams
    def initialize_teams_widgets(self, left_layout, right_layout):
        self.team1_layout = QGridLayout()
        self.team2_layout = QGridLayout()
        self.team1_group = self.create_group_widget("Team 1", self.team1_layout, left_layout)
        self.team2_group = self.create_group_widget("Team 2", self.team2_layout, right_layout)

        self.team1_boxes, self.team1_photos, self.team1_photo_frames = self.create_team_widgets(5)
        self.team2_boxes, self.team2_photos, self.team2_photo_frames = self.create_team_widgets(5)

        self.add_agent_widgets_to_layout(self.team1_boxes, self.team1_photos, self.team1_photo_frames,
                                         self.team1_layout)
        self.add_agent_widgets_to_layout(self.team2_boxes, self.team2_photos, self.team2_photo_frames,
                                         self.team2_layout)

    # Creates the QGroupBox object with the provided title and layout
    def create_group_widget(self, title, layout, parent_layout):
        group = QGroupBox(title, alignment=Qt.AlignCenter)
        group.setLayout(layout)
        parent_layout.addWidget(group)
        return group

    # Creates the widgets for each of the teams
    def create_team_widgets(self, count):
        boxes = [self.create_combo_box() for _ in range(count)]
        photos = [self.create_agent_photo_widget() for _ in range(count)]
        photo_frames = [self.create_agent_photo_frame() for _ in range(count)]
        return boxes, photos, photo_frames

    # Creates combo boxes for each agent on each team
    def create_combo_box(self):
        box = QComboBox()
        box.setMaximumHeight(self.MAX_BOX_HEIGHT)
        box.setMaximumWidth(self.MAX_BOX_WIDTH)
        box.addItem("")
        box.addItems(self.AGENT_OPTIONS)
        box.setCurrentIndex(0)
        box.currentIndexChanged.connect(self.update_agent_selections)
        return box

    # Creates widgets for displaying agent photos
    def create_agent_photo_widget(self):
        photo = QLabel()
        photo.setAlignment(Qt.AlignCenter)
        photo.setFixedSize(self.PHOTO_HEIGHT, self.PHOTO_HEIGHT)
        return photo

    # Creates frame to hold the agent photo widget
    def create_agent_photo_frame(self):
        frame = QFrame()
        frame.setMaximumHeight(self.PHOTO_HEIGHT)
        frame.setFrameStyle(QFrame.Box)
        frame.setLineWidth(1)
        frame.setMidLineWidth(0)
        return frame

    # Adds agent-related widgets to the layout
    def add_agent_widgets_to_layout(self, boxes, photos, photo_frames, layout):
        for index, (box, photo, frame) in enumerate(zip(boxes, photos, photo_frames)):
            agent_label = QLabel(f"Agent {index + 1}:")
            agent_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(agent_label, 0, index)
            layout.addWidget(box, 1, index)
            layout.addWidget(frame, 2, index)
            layout.addWidget(photo, 2, index)

    # Creates and configures both the rank and map selection widgets
    def setup_rank_and_map_widgets(self):
        rank_map_layout = QHBoxLayout()
        self.rank_group = QGroupBox("Rank", alignment=Qt.AlignCenter)
        self.rank_layout = QVBoxLayout(self.rank_group)
        self.map_group = QGroupBox("Map", alignment=Qt.AlignCenter)
        self.map_layout = QVBoxLayout(self.map_group)

        self.rank_image = QLabel()
        self.rank_image.setFixedSize(QSize(256, 256))

        self.map_image = QLabel()
        self.map_image.setFixedSize(QSize(450, 256))

        self.rank_box = QComboBox()
        self.rank_box.addItem("")
        self.rank_box.addItems(self.RANK_CATEGORIES)
        self.rank_box.setCurrentIndex(0)
        self.rank_box.setStyleSheet("height: 15px; width: 150px")
        self.rank_box.currentIndexChanged.connect(self.update_rank_photo)
        self.rank_layout.addWidget(self.rank_box, alignment=(Qt.AlignTop | Qt.AlignCenter))
        self.rank_layout.addWidget(self.rank_image, alignment=(Qt.AlignTop | Qt.AlignCenter))

        self.map_box = QComboBox()
        self.map_box.addItem("")
        self.map_box.addItems(self.RANKED_MAPS)
        self.map_box.setCurrentIndex(0)
        self.map_box.setStyleSheet("height: 15px; width: 150px")
        self.map_box.currentIndexChanged.connect(self.update_map_photo)
        self.map_layout.addWidget(self.map_box, alignment=(Qt.AlignTop | Qt.AlignCenter))
        self.map_layout.addWidget(self.map_image, alignment=Qt.AlignCenter)

        rank_map_layout.addWidget(self.rank_group)
        rank_map_layout.addWidget(self.map_group)
        self.rank_map_layout = rank_map_layout

    # Creates and configures widgets for displaying graphs
    def setup_graph_widgets(self):
        self.graph_layout = QHBoxLayout()

        self.figure1 = Figure(figsize=(4, 4))
        self.canvas1 = FigureCanvas(self.figure1)
        self.figure2 = Figure(figsize=(4, 4))
        self.canvas2 = FigureCanvas(self.figure2)
        self.figure3 = Figure(figsize=(4, 4))
        self.canvas3 = FigureCanvas(self.figure3)

        self.canvas1.setStyleSheet("QFrame {border: 1px solid black;}")
        self.canvas2.setStyleSheet("QFrame {border: 1px solid black;}")
        self.canvas3.setStyleSheet("QFrame {border: 1px solid black;}")

        frame_layout1 = QVBoxLayout()
        frame_layout1.addWidget(self.canvas1)
        self.frame1 = QFrame()
        self.frame1.setLayout(frame_layout1)
        self.graph_layout.addWidget(self.frame1)

        frame_layout2 = QVBoxLayout()
        frame_layout2.addWidget(self.canvas2)
        self.frame2 = QFrame()
        self.frame2.setLayout(frame_layout2)
        self.graph_layout.addWidget(self.frame2)

        frame_layout3 = QVBoxLayout()
        frame_layout3.addWidget(self.canvas3)
        self.frame3 = QFrame()
        self.frame3.setLayout(frame_layout3)
        self.graph_layout.addWidget(self.frame3)

    # Creates a button that the user can interact with
    def add_button(self, text, color, click_method):
        button = QPushButton(text)
        button.setStyleSheet(self.BUTTON_STYLESHEET.format(color))
        button.setSizePolicy(self.BUTTON_FIXED_POLICY)
        button.clicked.connect(click_method)
        return button

    # Creates and configures each of the button widgets in the application
    def setup_button_widgets(self):
        self.button_layout = QHBoxLayout()
        self.download_button = self.add_button('Download Latest Match Data ', '#007BFF',
                                               self.on_download_data_button_click)
        self.predict_button = self.add_button('Make a Prediction', '#007BFF', self.on_make_prediction_button_click)
        self.exit_button = self.add_button('Exit Application', '#6C757D', self.exit_program)

        self.button_layout.setSpacing(525)
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.download_button)
        self.button_layout.addWidget(self.predict_button)
        self.button_layout.addWidget(self.exit_button)
        self.button_layout.addStretch()

    # Updates the photo underneath the rank combo box to display the corresponding rank's logo
    def update_rank_photo(self):
        rank_name = self.rank_box.currentText()
        if rank_name != "":
            rank_name = rank_name.replace("/", "").lower()
            rank_photo_path = os.path.join(EXECUTABLE_DIRECTORY, "..", "..", "ImageAssets", "RankPhotos",
                                           f"{rank_name}.png")
            rank_pixmap = QPixmap(rank_photo_path)
            rank_pixmap = rank_pixmap.scaledToHeight(256)
            self.rank_image.setPixmap(rank_pixmap)
        else:
            self.rank_image.clear()

    # Updates the photo underneath the map combo box to display the corresponding map's photo
    def update_map_photo(self):
        map_name = self.map_box.currentText()
        if map_name != "":
            map_name = map_name.replace("/", "").lower()
            map_photo_path = os.path.join(EXECUTABLE_DIRECTORY, "..", "..", "ImageAssets/MapPhotos", f"{map_name}.png")
            map_pixmap = QPixmap(map_photo_path)
            map_pixmap = map_pixmap.scaledToHeight(256)
            self.map_image.setPixmap(map_pixmap)
        else:
            self.map_image.clear()

    # Creates and configures rank and map widgets
    @staticmethod
    def create_rank_and_map_widgets(rank_options, map_options, update_rank_image, update_map_image):
        rank_group = QGroupBox("Rank", alignment=Qt.AlignCenter)
        rank_layout = QVBoxLayout(rank_group)

        map_group = QGroupBox("Map", alignment=Qt.AlignCenter)
        map_layout = QVBoxLayout(map_group)

        rank_image = QLabel()
        rank_image.setFixedSize(QSize(256, 256))

        map_image = QLabel()
        map_image.setFixedSize(QSize(450, 256))

        rank_box = QComboBox()
        rank_box.addItem("")
        rank_box.addItems(rank_options)
        rank_box.setCurrentIndex(0)
        rank_box.setStyleSheet("height: 15px; width: 150px")
        rank_box.currentIndexChanged.connect(update_rank_image)
        rank_layout.addWidget(rank_box, alignment=(Qt.AlignTop | Qt.AlignCenter))
        rank_layout.addWidget(rank_image, alignment=(Qt.AlignTop | Qt.AlignCenter))

        map_box = QComboBox()
        map_box.addItem("")
        map_box.addItems(map_options)
        map_box.setCurrentIndex(0)
        map_box.setStyleSheet("height: 15px; width: 150px")
        map_box.currentIndexChanged.connect(update_map_image)
        map_layout.addWidget(map_box, alignment=(Qt.AlignTop | Qt.AlignCenter))
        map_layout.addWidget(map_image, alignment=Qt.AlignCenter)

        return rank_group, map_group

    # Updates the agent selections, choices, combo box lists, and photos
    def update_agent_selections(self):
        selected_agents1 = [box.currentText() for box in self.team1_boxes if box.currentText() != ""]
        selected_agents2 = [box.currentText() for box in self.team2_boxes if box.currentText() != ""]

        for i in range(5):
            if i >= len(selected_agents1):
                self.team1_photos[i].clear()

            if i >= len(selected_agents2):
                self.team2_photos[i].clear()

        # Iterate over all boxes and update their item lists
        for box_list, selected_agents, agent_photos in [
            (self.team1_boxes, selected_agents1, self.team1_photos),
            (self.team2_boxes, selected_agents2, self.team2_photos)
        ]:
            for box, agent_photo in zip(box_list, agent_photos):
                box.blockSignals(True)
                current_text = box.currentText()
                box.clear()
                box.addItem("")
                box.addItems(
                    [agent for agent in self.AGENT_OPTIONS if agent not in selected_agents or agent == current_text])
                box.setCurrentText(current_text)
                box.blockSignals(False)

                # Update agent photos
                if current_text != "":
                    agent_name = current_text.replace("/", "").lower()
                    if agent_name == "kayo":  # Special case for KAY/O
                        agent_name = "kayo"
                    photo_path = os.path.join(EXECUTABLE_DIRECTORY, "..", "..", "ImageAssets/AgentPhotos",
                                              f"{agent_name}.png")
                    pixmap = QPixmap(photo_path)
                    pixmap = pixmap.scaledToHeight(150)
                    agent_photo.setPixmap(pixmap)

    # Scrapes and organizes match data for each rank on each map
    def on_download_data_button_click(self):
        ranks_iron = [(3, "Iron 1"), (4, "Iron 2"), (5, "Iron 3")]
        ranks_bronze = [(6, "Bronze 1"), (7, "Bronze 2"), (8, "Bronze 3")]
        ranks_silver = [(9, "Silver 1"), (10, "Silver 2"), (11, "Silver 3")]
        ranks_gold = [(12, "Gold 1"), (13, "Gold 2"), (14, "Gold 3")]
        ranks_platinum = [(15, "Platinum 1"), (16, "Platinum 2"), (17, "Platinum 3")]
        ranks_diamond = [(18, "Diamond 1"), (19, "Diamond 2"), (20, "Diamond 3")]
        ranks_ascendant = [(21, "Ascendant 1"), (22, "Ascendant 2"), (23, "Ascendant 3")]
        ranks_immortal = [(24, "Immortal 1"), (25, "Immortal 2"), (26, "Immortal 3")]
        ranks_radiant = [(27, "Radiant")]
        ranked_maps = ["Split", "Ascent", "Haven", "Bind", "Fracture", "Pearl", "Lotus"]

        self.download_button.setDisabled(True)

        QMessageBox.information(self.main_window, "Information",
                                "Downloading the newest competitive match data! This can take a few minutes...")

        try:
            scrape_data(ranks_iron + ranks_bronze + ranks_silver + ranks_gold + ranks_platinum + ranks_diamond +
                        ranks_ascendant + ranks_immortal + ranks_radiant, ranked_maps)

            organize_data_files(os.path.join(EXECUTABLE_DIRECTORY, "..", ".."))

            self.download_button.setDisabled(False)
            QMessageBox.information(self.main_window, "Information", "The newest available competitive match data has "
                                                                     "completed downloading.")

        except Exception as e:
            QMessageBox.critical(self.main_window, "Error", f"An error occurred: {str(e)}")

    # Makes prediction based on selected agent, map, and rank. Displays the prediction/statistics via graphs,
    # and a message box
    def on_make_prediction_button_click(self):
        try:
            team1_agents, team2_agents, rank, map_name = self.get_inputs()
            self.validate_inputs(team1_agents, team2_agents, rank, map_name)
            self.data[(rank, map_name)] = load_data(rank, map_name, os.path.join(EXECUTABLE_DIRECTORY, "..", "..", ))
            self.clear_figures()

            result_string, team1_prob, team2_prob, agent_pick_rate, agent_win_rate = self.get_prediction_and_win_rates(
                rank,
                map_name,
                team1_agents,
                team2_agents)

            self.draw_graphs(team1_prob, team2_prob, team1_agents, team2_agents, agent_pick_rate,
                             agent_win_rate)

            self.show_prediction_result(result_string)

        except Exception as e:
            self.show_error_message(str(e))

    # Gets the inputs for agents selected on each team, the rank, and the map
    def get_inputs(self):
        team1_agents = [box.currentText() for box in self.team1_boxes if box.currentText() != ""]
        team2_agents = [box.currentText() for box in self.team2_boxes if box.currentText() != ""]
        rank = self.rank_box.currentText()
        map_name = self.map_box.currentText()
        return team1_agents, team2_agents, rank, map_name

    @staticmethod
    def validate_inputs(team1_agents, team2_agents, rank, map_name):
        if not rank:
            raise ValueError("Please select a rank.")
        if not map_name:
            raise ValueError("Please select a map.")
        if len(team1_agents) < 5 or len(team2_agents) < 5:
            raise ValueError("Please select all 5 agents for both Team 1 and Team 2.")

    @staticmethod
    def show_error_message(text):
        msg = QMessageBox()
        msg.setWindowTitle("Error")
        msg.setText(text)
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()

    def clear_figures(self):
        self.figure1.clear()
        self.figure2.clear()
        self.figure3.clear()

    # Gets the winning team prediction and probabilities of each team winning
    # Gets the pick rate and win rate of each agent on each team
    @staticmethod
    def get_prediction_and_win_rates(rank, map_name, team1_agents, team2_agents):
        winning_team, team1_prob, team2_prob = get_prediction(rank, map_name, team1_agents, team2_agents)

        agent_pick_rate = {
            agent: get_pick_rate(agent, rank, map_name, team1_agents + team2_agents) for agent in
            MainWindow.AGENT_OPTIONS
        }
        agent_win_rate = {
            agent: get_win_rate(agent, rank, map_name, team1_agents + team2_agents) for agent in
            MainWindow.AGENT_OPTIONS
        }
        return winning_team, team1_prob, team2_prob, agent_pick_rate, agent_win_rate

    # Creates three separate graphs for displaying data to the user
    # Plots the data on each graph, sets titles and labels, and updates the canvases
    def draw_graphs(self, team1_prob, team2_prob, team1_agents, team2_agents, agent_pick_rate,
                    agent_win_rate):
        team1_pick_rates = [agent_pick_rate[agent] for agent in team1_agents]
        team2_pick_rates = [agent_pick_rate[agent] for agent in team2_agents]
        team1_win_rates = [agent_win_rate[agent] for agent in team1_agents]
        team2_win_rates = [agent_win_rate[agent] for agent in team2_agents]

        ax1 = self.figure1.add_subplot(111)
        ax2 = self.figure2.add_subplot(111)
        ax3 = self.figure3.add_subplot(111)

        ax1.pie([team1_prob, team2_prob], labels=["Team 1", "Team 2"], autopct='%1.1f%%', colors=['#507DBC', '#F08A4B'])
        ax2.bar(range(len(team1_agents)), team1_pick_rates, color='#507DBC', label='Team 1')
        ax2.bar(range(len(team1_agents), len(team1_agents) + len(team2_agents)), team2_pick_rates, color='#F08A4B',
                label='Team 2')
        ax3.bar(range(len(team1_agents)), team1_win_rates, color='#507DBC', label='Team 1')
        ax3.bar(range(len(team1_agents), len(team1_agents) + len(team2_agents)), team2_win_rates, color='#F08A4B',
                label='Team 2')

        self.figure1.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8)
        self.figure2.subplots_adjust(left=0.3, bottom=0.3, right=0.8, top=0.8)
        self.figure3.subplots_adjust(left=0.3, bottom=0.3, right=0.8, top=0.8)

        ax1.set_title('Team Win Chances', fontsize=16, fontweight='bold')
        ax1.axis('equal')

        ax2.set_title('Agent Pick Rates', fontsize=16, fontweight='bold')
        ax2.set_xticks(range(len(team1_agents) + len(team2_agents)))
        ax2.set_xticklabels(team1_agents + team2_agents, rotation=45, ha='right')
        ax2.set_xlabel('Agents', fontsize=12)
        ax2.set_ylabel('Pick Rate', fontsize=12)

        ax3.set_title('Agent Win Rates', fontsize=16, fontweight='bold')
        ax3.set_xticks(range(len(team1_agents) + len(team2_agents)))
        ax3.set_xticklabels(team1_agents + team2_agents, rotation=45, ha='right')
        ax3.set_xlabel('Agents', fontsize=12)
        ax3.set_ylabel('Win Rate', fontsize=12)

        self.canvas1.draw_idle()
        self.canvas2.draw_idle()
        self.canvas3.draw_idle()

    @staticmethod
    def show_prediction_result(result_string):
        msg = QMessageBox()
        msg.setWindowTitle("Prediction Result")
        msg.setText(result_string)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

    @staticmethod
    def exit_program():
        sys.exit()


app = MainWindow(sys.argv)
sys.exit(app.exec_())
