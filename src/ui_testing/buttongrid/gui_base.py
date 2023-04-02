

import copy
import string
import tkinter as tk
from itertools import groupby


LARGE_FONT = ("Verdana", 12)
VLF = very_large_font = 'Helvetica 56 bold'

HOME_ID = 'AAA'


class MainFrame(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.frame_visit_history = [HOME_ID]

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        letters = string.ascii_uppercase + string.ascii_lowercase  # letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + 'abcdefghijklmnopqrstuvwxyz'
        letters = letters.replace('l', '')  # remove lower case l because it is exactly the same letter as capital i
        self.labels = letters
        print(len(self.labels), 'len(self.labels')

        self.labels = self.labels[:40]
        self.labels = [x * 3 for x in self.labels]

        self.identifier_list = copy.deepcopy(self.labels)

        edges_per_node = 3
        num_levels = 4

        self.assocs = []
        prev_level_nodes = []

        for i in range(num_levels):
            nodes_on_level = edges_per_node ** i
            labels_for_current_level = self.labels[:nodes_on_level]
            self.labels = self.labels[nodes_on_level:]

            group_idx = -1
            idx = -1
            for j, x in enumerate(labels_for_current_level):

                if x == HOME_ID:
                    prev_level_nodes.append(x)
                    break

                if j % edges_per_node == 0:
                    group_idx += 1
                    self.assocs.append([prev_level_nodes[group_idx]])

                idx += 1
                self.assocs[-1].append(labels_for_current_level[idx])
                if j == len(labels_for_current_level) -1:
                    prev_level_nodes = labels_for_current_level

        self.frames = {}
        for idn in self.identifier_list:

            sel_assocs = None
            for x in self.assocs:
                if x[0] == idn:
                    sel_assocs = x[1:]

            frame = MyFrame(container, self, idn, sel_assocs)
            self.frames[idn] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.select_frame(HOME_ID, HOME_ID)

    def select_frame(self, came_from_id, go_to_id):
        if go_to_id is not None:
            sel_frame = self.frames[go_to_id]
            sel_frame.came_from_id = came_from_id
            sel_frame.tkraise()
            if go_to_id == HOME_ID:
                self.frame_visit_history = [HOME_ID]
            self.frame_visit_history.append(go_to_id)

    def roll_back(self, current_frame_id):
        self.frame_visit_history = [x[0] for x in groupby(self.frame_visit_history)]
        go_to_id = self.frame_visit_history[-1]
        del self.frame_visit_history[-1]
        if go_to_id == current_frame_id:
            go_to_id = self.frame_visit_history[-1]
        sel_frame = self.frames[go_to_id]
        sel_frame.tkraise()
        if len(self.frame_visit_history) <= 1:
            self.frame_visit_history = [HOME_ID]


class MyFrame(tk.Frame):

    def __init__(self, parent, controller, identifier, association_ids=None):
        tk.Frame.__init__(self, parent)

        self.parent = parent
        self.controller = controller
        self.identifier = identifier
        self.came_from_id = HOME_ID
        if association_ids is None:
            self.association_ids = [None, None, None]
        else:
            self.association_ids = association_ids

        home_button = tk.Button(self, text="Home", width=20, height=10, command=lambda: controller.select_frame(self.identifier, HOME_ID))
        home_button.grid(row=0, column=0, padx=5, pady=5)

        # back_button = tk.Button(self, text="Back", width=20, height=10, command=lambda: controller.select_frame(self.identifier, self.came_from_id))
        back_button = tk.Button(self, text="Back", width=20, height=10, command=lambda: controller.roll_back(self.identifier))
        back_button.grid(row=1, column=0, padx=5, pady=5)

        dummy_button = tk.Button(self, text="Dummy", width=20, height=10)
        dummy_button.grid(row=2, column=0, padx=5, pady=5)

        button1 = tk.Button(self, text="Button 1", width=20, height=10, command=lambda: controller.select_frame(self.identifier, self.association_ids[0]))
        button1.grid(row=0, column=1, padx=5, pady=5)

        button2 = tk.Button(self, text="Button 2", width=20, height=10, command=lambda: controller.select_frame(self.identifier, self.association_ids[1]))
        button2.grid(row=1, column=1, padx=5, pady=5)

        button3 = tk.Button(self, text="Button 3", width=20, height=10, command=lambda: controller.select_frame(self.identifier, self.association_ids[2]))
        button3.grid(row=2, column=1, padx=5, pady=5)

        label = tk.Label(self, text=str(self.identifier[0]), font=VLF, width=3, height=1)
        label.grid(row=0, column=2, padx=5, pady=5)

        label = tk.Label(self, text=str(self.identifier[1]), font=VLF, width=3, height=1)
        label.grid(row=1, column=2, padx=5, pady=5)

        label = tk.Label(self, text=str(self.identifier[2]), font=VLF, width=3, height=1)
        label.grid(row=2, column=2, padx=5, pady=5)


if __name__ == '__main__':
    app = MainFrame()
    app.title('ButtonGrid')
    app.geometry('480x520+10+20')
    app.mainloop()

