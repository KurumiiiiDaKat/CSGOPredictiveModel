from compare_teams import cache_teams, compare_teams
import tkinter as tk

maps = {"Vertigo": 'vtg',
        "Anubis": 'anb',
        "Dust II": 'd2',
        "Ancient": 'anc',
        "Mirage": 'mrg',
        "Cobble": 'cbl',
        "Inferno": 'inf',
        "Overpass": 'ovp',
        "Train": 'trn',
        "Cache": 'cch',
        "Nuke": 'nuke'}

teams = {}

class Application(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # Grid size and initialization.
        grid_size = (20, 5)
        for i in range(grid_size[0]): self.rowconfigure(i, weight=1)
        for j in range(grid_size[1]): self.columnconfigure(j, weight=1)

        # First team selection menu.
        self.all_teams = list(teams.keys())
        def update_team1_list(*args):
            search = current_t1_search.get().lower()
            if not search: filtered = self.all_teams
            else: filtered = [t for t in self.all_teams if search in t.lower()]

            t1_listbox.delete(0, tk.END)
            for team in filtered: t1_listbox.insert(tk.END, team)

        def team1_new_team(event):
            t1_current_index = t1_listbox.curselection()
            t1_current = t1_listbox.get(t1_current_index)
            current_t2_search.trace_vdelete("w", current_t2_search.trace_id)
            current_t1_search.set(t1_current)
            current_t2_search.trace_id = current_t2_search.trace_add("write", update_team2_list)

        current_t1_search = tk.StringVar(self)
        current_t1_search.trace_id = current_t1_search.trace_add("write", update_team1_list)
        t1_search = tk.Entry(self, textvariable=current_t1_search)
        t1_search.grid(row=4, column=1)

        t1_initial_team_list = tk.Variable(value=self.all_teams)
        t1_listbox = tk.Listbox(self, listvariable=t1_initial_team_list, height=5, selectmode=tk.SINGLE)
        t1_listbox.bind('<<ListboxSelect>>', team1_new_team)
        t1_listbox.grid(row=5, column=1)

        # Second team selection menu.
        def update_team2_list(*args):
            search = current_t2_search.get().lower()
            if not search: filtered = self.all_teams
            else: filtered = [t for t in self.all_teams if search in t.lower()]

            t2_listbox.delete(0, tk.END)
            for team in filtered: t2_listbox.insert(tk.END, team)

        def team2_new_team(event):
            t2_current_index = t2_listbox.curselection()
            t2_current = t2_listbox.get(t2_current_index)
            current_t1_search.trace_vdelete("w", current_t1_search.trace_id)
            current_t2_search.set(t2_current)
            current_t1_search.trace_id = current_t1_search.trace_add("write", update_team1_list)

        current_t2_search = tk.StringVar(self)
        current_t2_search.trace_id = current_t2_search.trace_add("write", update_team2_list)
        t2_search = tk.Entry(self, textvariable=current_t2_search)
        t2_search.grid(row=4, column=2)

        t2_initial_team_list = tk.Variable(value=self.all_teams)
        t2_listbox = tk.Listbox(self, listvariable=t2_initial_team_list, height=5, selectmode=tk.SINGLE)
        t2_listbox.bind('<<ListboxSelect>>', team2_new_team)
        t2_listbox.grid(row=5, column=2)

        # Map selection drop down menu.
        current_map = tk.StringVar(self)
        current_map.set("Select Map")
        map_list = tk.OptionMenu(self, current_map, *list(maps.keys()))
        map_list.grid(row=4, column=3)

        # Submit button.
        def press_submit():
            t1 = current_t1_search.get()
            if t1 not in self.all_teams:
                error_message_content.set("Invalid team 1.")
                return
            t2 = current_t2_search.get()
            if t2 not in self.all_teams:
                error_message_content.set("Invalid team 2.")
                return
            if t1 == t2:
                error_message_content.set("Teams cannot be the same.")
                return
            map = current_map.get()
            if map == "Select Map" or map not in maps:
                error_message_content.set("Invalid map.")
                return

            map = maps[map]     # Convert to short name
            result = compare_teams(t1, t2, map)
            winning_team = t1 if result[0] == 1 else t2
            error_message_content.set(winning_team + " wins this match.")

        submit_button = tk.Button(self, command=press_submit, text="Compare", width=8, height=2)
        submit_button.grid(row=10, column=2)

        # Error message.
        error_message_content = tk.StringVar(self)
        error_message_content.set("")
        error_message = tk.Label(self, textvariable=error_message_content)
        error_message.grid(row=11, column=2)

if __name__ == "__main__":
    teams = cache_teams()

    root = tk.Tk()
    Application(root).pack(side="top", fill="both", expand=True)
    root.geometry("1000x600")
    root.title("CS:GO Match Predictive Model")
    root.mainloop()
