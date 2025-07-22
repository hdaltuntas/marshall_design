# --- START OF FILE marshall_tk.py ---
"""
A Tkinter-based desktop application for Marshall asphalt mix design analysis.

This program allows users to input laboratory data for different asphalt mix
types, calculates key volumetric and mechanical properties, determines the
Optimum Binder Content (OBC), and visualizes the results on graphs. The user
interface supports multiple languages (English and Turkish).
"""

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==============================================================================
# CONSTANTS
# ==============================================================================

# All UI text is stored here, allowing for easy translation.
LANGUAGES = {
    "TR": {
        "title": "Marshall Asfalt Karışım Tasarım Programı",
        "design_type_frame": "Tasarım Tipi",
        "select_layer_type": "Tabaka Tipini Seçin:",
        "general_info_frame": "Genel Bilgiler",
        "gsb_label": "Agrega Özgül Ağırlığı (Gsb):",
        "target_va_label": "Hedef Hava Boşluğu (%Va):",
        "specs_frame": "Şartname Limitleri",
        "prop_col": "Özellik",
        "min_col": "Min",
        "max_col": "Max",
        "lab_data_frame": "Laboratuvar Verileri",
        "bitumen_col": "Bitüm (%)",
        "stability_col": "Stabilite",
        "flow_col": "Akma (mm)",
        "calc_button": "Hesapla ve Grafikleri Çiz",
        "results_frame": "Sonuçlar",
        "press_button_to_calc": "Hesaplama yapmak için butona basın...",
        "error_title": "Hata",
        "value_error_msg": (
            "Lütfen tablodaki tüm alanları doldurun ve sayısal değerler girin."
        ),
        "insufficient_data_msg": (
            "Hesaplama için en az 3 geçerli veri satırı gereklidir."
        ),
        "spec_error_title": "Şartname Hatası",
        "spec_error_msg": "'{key}' için girilen limitler sayısal olmalıdır.",
        "results_design_type": "Tasarım Tipi",
        "results_obi": "OPTIMUM BİTÜM İÇERİĞİ (OBİ)",
        "results_based_on_va": (
            "(%{va:.1f} Hava Boşluğuna göre belirlenmiştir)"
        ),
        "results_status": "Durum",
        "results_prop_header": "Özellik",
        "results_value_header": "Değer",
        "results_spec_header": "Şartname",
        "status_ok": "UYGUN",
        "status_fail_min": "UYGUN DEĞİL (Min!)",
        "status_fail_max": "UYGUN DEĞİL (Max!)",
        "prop_names": {
            'Va': 'Hava Boşluğu (%)', 'Stabilite': 'Stabilite (kg)',
            'Akma': 'Akma (mm)', 'VMA': 'VMA (%)', 'VFA': 'VFA (%)'
        },
        "plot_bitumen": "Bitüm İçeriği (%)",
        "plot_stability": "Stabilite",
        "plot_unit_weight": "Birim Hacim Ağ. (kg/m³)",
        "plot_va": "Hava Boşluğu (Va) (%)",
        "plot_flow": "Akma (mm)",
        "plot_vma": "VMA (%)",
        "plot_vfa": "VFA (%)"
    },
    "EN": {
        "title": "Marshall Asphalt Mix Design Program",
        "design_type_frame": "Design Type",
        "select_layer_type": "Select Layer Type:",
        "general_info_frame": "General Information",
        "gsb_label": "Aggregate Specific Gravity (Gsb):",
        "target_va_label": "Target Air Voids (%Va):",
        "specs_frame": "Specification Limits",
        "prop_col": "Property",
        "min_col": "Min",
        "max_col": "Max",
        "lab_data_frame": "Laboratory Data",
        "bitumen_col": "Bitumen (%)",
        "stability_col": "Stability",
        "flow_col": "Flow (mm)",
        "calc_button": "Calculate and Plot Graphs",
        "results_frame": "Results",
        "press_button_to_calc": "Press the button to perform calculation...",
        "error_title": "Error",
        "value_error_msg": (
            "Please fill all fields in the table and enter numeric values."
        ),
        "insufficient_data_msg": (
            "At least 3 valid data rows are required for calculation."
        ),
        "spec_error_title": "Specification Error",
        "spec_error_msg": "Limits entered for '{key}' must be numeric.",
        "results_design_type": "Design Type",
        "results_obi": "OPTIMUM BINDER CONTENT (OBC)",
        "results_based_on_va": (
            "(Determined based on {va:.1f}% Air Voids)"
        ),
        "results_status": "Status",
        "results_prop_header": "Property",
        "results_value_header": "Value",
        "results_spec_header": "Specification",
        "status_ok": "OK",
        "status_fail_min": "NOT OK (Min!)",
        "status_fail_max": "NOT OK (Max!)",
        "prop_names": {
            'Va': 'Air Voids (%)', 'Stabilite': 'Stability (kg)',
            'Akma': 'Flow (mm)', 'VMA': 'VMA (%)', 'VFA': 'VFA (%)'
        },
        "plot_bitumen": "Bitumen Content (%)",
        "plot_stability": "Stability",
        "plot_unit_weight": "Unit Weight (kg/m³)",
        "plot_va": "Air Voids (Va) (%)",
        "plot_flow": "Flow (mm)",
        "plot_vma": "VMA (%)",
        "plot_vfa": "VFA (%)"
    }
}

DESIGN_CRITERIA = {
    "wearing": {
        "tr": "Aşınma", "en": "Wearing Course", "hedef_va": 4.0,
        "specs": {
            'Va': ('3.0', '5.0'), 'Stabilite': ('900', ''),
            'Akma': ('2.0', '4.0'), 'VMA': ('14', '16'), 'VFA': ('65', '75')
        }
    },
    "binder": {
        "tr": "Binder", "en": "Binder Course", "hedef_va": 5.0,
        "specs": {
            'Va': ('4.0', '6.0'), 'Stabilite': ('750', ''),
            'Akma': ('2.0', '4.0'), 'VMA': ('13', '15'), 'VFA': ('60', '75')
        }
    },
    "base": {
        "tr": "Bitümlü Temel", "en": "Bituminous Base", "hedef_va": 5.0,
        "specs": {
            'Va': ('4.0', '6.0'), 'Stabilite': ('600', ''),
            'Akma': ('2.0', '5.0'), 'VMA': ('12', '14.5'), 'VFA': ('55', '75')
        }
    }
}


class MarshallApp(tk.Tk):
    """
    The main application class for the Marshall Mix Design program.

    This class builds the GUI, handles user interactions, and performs all
    necessary calculations and plotting.
    """

    def __init__(self):
        """Initialize the application and its main components."""
        super().__init__()
        self.geometry("1250x800")

        self.current_lang = "TR"  # Default language
        self.texts = LANGUAGES[self.current_lang]

        # --- Main UI Frames ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill="both", expand=True)

        self._setup_language_selector(main_frame)

        left_frame = ttk.Frame(main_frame, width=450)
        left_frame.pack(side="left", fill="y", padx=(0, 10))
        left_frame.pack_propagate(False)
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True)

        self.create_input_widgets(left_frame)
        self.create_graph_widgets(right_frame)

        self.update_ui_texts()  # Populate the UI for the first time

    def _setup_language_selector(self, parent):
        """Create and pack the language selection combobox."""
        top_bar = ttk.Frame(parent)
        top_bar.pack(side="top", fill="x", pady=(0, 10))
        ttk.Label(top_bar, text="Dil / Language:").pack(side="left", padx=(0, 5))
        self.lang_combo = ttk.Combobox(
            top_bar, values=["TR", "EN"], width=5, state="readonly"
        )
        self.lang_combo.pack(side="left")
        self.lang_combo.set(self.current_lang)
        self.lang_combo.bind("<<ComboboxSelected>>", self.change_language)

    def create_input_widgets(self, parent):
        """Create all the input widgets on the left panel."""
        self._create_design_type_widgets(parent)
        self._create_info_widgets(parent)
        self._create_spec_widgets(parent)
        self._create_lab_data_widgets(parent)
        self._create_control_widgets(parent)

    def _create_design_type_widgets(self, parent):
        """Create widgets for selecting the design type."""
        self.design_type_frame = ttk.LabelFrame(parent, padding="10")
        self.design_type_frame.pack(fill="x", pady=(0, 5))
        self.select_layer_label = ttk.Label(self.design_type_frame)
        self.select_layer_label.pack(side="left", padx=5)
        self.design_type_combo = ttk.Combobox(
            self.design_type_frame, state="readonly"
        )
        self.design_type_combo.pack(side="left", fill="x", expand=True)
        self.design_type_combo.bind(
            "<<ComboboxSelected>>", self.on_design_type_change
        )

    def _create_info_widgets(self, parent):
        """Create widgets for general mix information (Gsb, Target Va)."""
        self.info_frame = ttk.LabelFrame(parent, padding="10")
        self.info_frame.pack(fill="x", pady=5)
        self.gsb_label = ttk.Label(self.info_frame)
        self.gsb_label.grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.gsb_entry = ttk.Entry(self.info_frame, width=12)
        self.gsb_entry.insert(0, "2.65")
        self.gsb_entry.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        self.target_va_label = ttk.Label(self.info_frame)
        self.target_va_label.grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.target_va_entry = ttk.Entry(self.info_frame, width=12, state="readonly")
        self.target_va_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)

    def _create_spec_widgets(self, parent):
        """Create the table for entering specification limits."""
        self.spec_frame = ttk.LabelFrame(parent, padding="10")
        self.spec_frame.pack(fill="x", pady=5)
        self.spec_entries = {}
        self.prop_label = ttk.Label(self.spec_frame)
        self.prop_label.grid(row=0, column=0, sticky="w")
        self.min_label = ttk.Label(self.spec_frame)
        self.min_label.grid(row=0, column=1, sticky="w")
        self.max_label = ttk.Label(self.spec_frame)
        self.max_label.grid(row=0, column=2, sticky="w")

        self.spec_labels = {}
        spec_keys = ['Va', 'Stabilite', 'Akma', 'VMA', 'VFA']
        for i, key in enumerate(spec_keys):
            label = ttk.Label(self.spec_frame)
            label.grid(row=i + 1, column=0, sticky="w", padx=5)
            self.spec_labels[key] = label
            min_entry = ttk.Entry(self.spec_frame, width=8)
            min_entry.grid(row=i + 1, column=1, padx=2)
            max_entry = ttk.Entry(self.spec_frame, width=8)
            max_entry.grid(row=i + 1, column=2, padx=2)
            self.spec_entries[key] = (min_entry, max_entry)

    def _create_lab_data_widgets(self, parent):
        """Create the table for entering laboratory data."""
        self.lab_data_frame = ttk.LabelFrame(parent, padding="10")
        self.lab_data_frame.pack(fill="x", pady=5)
        self.lab_data_headers = {}
        headers_keys = [
            'bitumen_col', 'Gmb', 'Gmm', 'stability_col', 'flow_col'
        ]
        for i, key in enumerate(headers_keys):
            label = ttk.Label(
                self.lab_data_frame, font=('Helvetica', 9, 'bold')
            )
            label.grid(row=0, column=i, padx=5, pady=5)
            self.lab_data_headers[key] = label

        self.table_entries = []
        sample_data = [
            [4.0, 2.345, 2.480, 950, 2.5], [4.5, 2.368, 2.460, 1100, 2.8],
            [5.0, 2.382, 2.441, 1150, 3.2], [5.5, 2.375, 2.422, 1080, 3.8],
            [6.0, 2.360, 2.404, 980, 4.5]
        ]
        for r in range(6):
            row_entries = []
            for c, _ in enumerate(headers_keys):
                entry = ttk.Entry(self.lab_data_frame, width=10)
                if r < len(sample_data):
                    entry.insert(0, str(sample_data[r][c]))
                entry.grid(row=r + 1, column=c, padx=2, pady=2)
                row_entries.append(entry)
            self.table_entries.append(row_entries)

    def _create_control_widgets(self, parent):
        """Create the calculate button and results text area."""
        self.calc_button = ttk.Button(
            parent, command=self.perform_calculation
        )
        self.calc_button.pack(fill="x", pady=5)
        self.results_frame = ttk.LabelFrame(parent, padding="10")
        self.results_frame.pack(fill="both", expand=True, pady=(5, 0))
        self.results_text = tk.Text(
            self.results_frame, height=10, width=55, wrap="word",
            font=('Courier', 9)
        )
        self.results_text.pack(fill="both", expand=True)
        self.results_text.config(state="disabled")

    def create_graph_widgets(self, parent):
        """Create the matplotlib figure and canvas on the right panel."""
        self.fig = plt.Figure(figsize=(7, 7), dpi=100)
        self.axs = self.fig.subplots(3, 2)
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def update_ui_texts(self):
        """Update all texts in the UI based on the selected language."""
        self.texts = LANGUAGES[self.current_lang]
        self.title(self.texts["title"])

        # Update frame titles
        self.design_type_frame.config(text=self.texts["design_type_frame"])
        self.info_frame.config(text=self.texts["general_info_frame"])
        self.spec_frame.config(text=self.texts["specs_frame"])
        self.lab_data_frame.config(text=self.texts["lab_data_frame"])
        self.results_frame.config(text=self.texts["results_frame"])

        # Update labels
        self.select_layer_label.config(text=self.texts["select_layer_type"])
        self.gsb_label.config(text=self.texts["gsb_label"])
        self.target_va_label.config(text=self.texts["target_va_label"])

        # Update table headers
        self.prop_label.config(text=self.texts["prop_col"])
        self.min_label.config(text=self.texts["min_col"])
        self.max_label.config(text=self.texts["max_col"])
        for key, label in self.spec_labels.items():
            label.config(text=self.texts["prop_names"][key])

        for key, label in self.lab_data_headers.items():
            label.config(text=self.texts.get(key, key))

        # Update button and results text placeholder
        self.calc_button.config(text=self.texts["calc_button"])
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert("1.0", self.texts["press_button_to_calc"])
        self.results_text.config(state="disabled")

        # Repopulate the design type combobox and maintain selection
        current_selection_key = self.get_selected_design_key(
            from_display_name=self.design_type_combo.get()
        )
        lang_key = 'tr' if self.current_lang == "TR" else 'en'
        display_values = [v[lang_key] for v in DESIGN_CRITERIA.values()]
        self.design_type_combo['values'] = display_values
        self.design_type_combo.set(
            DESIGN_CRITERIA[current_selection_key][lang_key]
        )

        self.on_design_type_change()
        self.perform_calculation(update_only=True)

    def change_language(self, event=None):
        """Trigger when the language combobox selection changes."""
        new_lang = self.lang_combo.get()
        if self.current_lang != new_lang:
            self.current_lang = new_lang
            self.update_ui_texts()

    def get_selected_design_key(self, from_display_name=None):
        """Find the internal key from the displayed name."""
        if not from_display_name:
            from_display_name = self.design_type_combo.get()

        if not from_display_name:
            return list(DESIGN_CRITERIA.keys())[0]

        # Check against both languages to find the key
        for key, data in DESIGN_CRITERIA.items():
            if data['tr'] == from_display_name or data['en'] == from_display_name:
                return key
        return list(DESIGN_CRITERIA.keys())[0]  # Fallback

    def on_design_type_change(self, event=None):
        """Trigger when the design type combobox selection changes."""
        selected_key = self.get_selected_design_key()
        criteria = DESIGN_CRITERIA[selected_key]

        self.target_va_entry.config(state="normal")
        self.target_va_entry.delete(0, tk.END)
        self.target_va_entry.insert(0, str(criteria["hedef_va"]))
        self.target_va_entry.config(state="readonly")

        for key, (min_entry, max_entry) in self.spec_entries.items():
            min_val, max_val = criteria["specs"].get(key, ('', ''))
            min_entry.delete(0, tk.END)
            min_entry.insert(0, min_val)
            max_entry.delete(0, tk.END)
            max_entry.insert(0, max_val)

    def get_specs_from_gui(self):
        """Retrieve and validate specification limits from the UI."""
        specs = {}
        for key, (min_entry, max_entry) in self.spec_entries.items():
            try:
                min_val = float(min_entry.get()) if min_entry.get() else None
                max_val = float(max_entry.get()) if max_entry.get() else None
                specs[key] = (min_val, max_val)
            except ValueError:
                msg = self.texts["spec_error_msg"].format(key=key)
                messagebox.showerror(self.texts["spec_error_title"], msg)
                return None
        return specs

    def _check_spec(self, value, min_spec, max_spec):
        """Helper to check a value against its min/max specifications."""
        if min_spec is not None and value < min_spec:
            return self.texts["status_fail_min"]
        if max_spec is not None and value > max_spec:
            return self.texts["status_fail_max"]
        return self.texts["status_ok"]

    def perform_calculation(self, update_only=False):
        """The main calculation and plotting workflow."""
        try:
            gsb = float(self.gsb_entry.get())
            target_air_voids = float(self.target_va_entry.get())
            data = []
            for row_entries in self.table_entries:
                if all(e.get() == "" for e in row_entries):
                    continue
                row_data = [float(e.get()) for e in row_entries]
                data.append(
                    {'Pb': row_data[0], 'Gmb': row_data[1],
                     'Gmm': row_data[2], 'Stabilite': row_data[3],
                     'Akma': row_data[4]}
                )

            if len(data) < 3:
                if not update_only:
                    messagebox.showerror(
                        self.texts["error_title"],
                        self.texts["insufficient_data_msg"]
                    )
                return
            specs = self.get_specs_from_gui()
            if specs is None:
                return
        except (ValueError, IndexError):
            if not update_only:
                messagebox.showerror(
                    self.texts["error_title"], self.texts["value_error_msg"]
                )
            return

        calculated_data = self._calculate_volumetrics(data, gsb)

        pb_list, stability_list, unit_weight_list, va_list, flow_list, \
            vma_list, vfa_list = [np.array([v[k] for v in calculated_data])
                                  for k in ['Pb', 'Stabilite', 'UnitWeight',
                                            'Va', 'Akma', 'VMA', 'VFA']]

        obc, obc_values = self._calculate_obc_and_properties(
            pb_list, stability_list, unit_weight_list, va_list,
            flow_list, vma_list, vfa_list, target_air_voids
        )

        self._update_plots(
            pb_list, stability_list, unit_weight_list, va_list,
            flow_list, vma_list, vfa_list, obc, obc_values
        )
        self._display_results(obc, obc_values, specs, target_air_voids)

    def _calculate_volumetrics(self, data, gsb):
        """Perform volumetric calculations for each data row."""
        calculated_data = []
        for row in data:
            pb = row['Pb']
            gmb = row['Gmb']
            gmm = row['Gmm']
            ps = 100 - pb
            va = 100 * (gmm - gmb) / gmm
            vma = 100 - (gmb * ps / gsb)
            vfa = 100 * (vma - va) / vma if vma > 0 else 0
            unit_weight = gmb * 1000
            calculated_data.append(
                {**row, 'Va': va, 'VMA': vma, 'VFA': vfa,
                 'UnitWeight': unit_weight}
            )
        calculated_data.sort(key=lambda x: x['Pb'])
        return calculated_data

    def _calculate_obc_and_properties(self, pb_list, stability_list,
                                      unit_weight_list, va_list, flow_list,
                                      vma_list, vfa_list, target_air_voids):
        """Determine OBC and calculate all property values at OBC."""
        va_poly = np.poly1d(np.polyfit(pb_list, va_list, 2))
        roots = [r.real for r in (va_poly - target_air_voids).roots
                 if r.imag == 0 and min(pb_list) < r.real < max(pb_list)]
        obc = roots[0] if roots else np.interp(
            target_air_voids, va_list[::-1], pb_list[::-1]
        )

        polys = {
            'Stabilite': np.poly1d(np.polyfit(pb_list, stability_list, 2)),
            'UnitWeight': np.poly1d(np.polyfit(pb_list, unit_weight_list, 2)),
            'Va': va_poly,
            'Akma': np.poly1d(np.polyfit(pb_list, flow_list, 2)),
            'VMA': np.poly1d(np.polyfit(pb_list, vma_list, 2)),
            'VFA': np.poly1d(np.polyfit(pb_list, vfa_list, 2))
        }
        obc_values = {key: poly(obc) for key, poly in polys.items()}
        return obc, obc_values

    def _update_plots(self, pb_list, stability_list, unit_weight_list,
                      va_list, flow_list, vma_list, vfa_list, obc,
                      obc_values):
        """Refresh all six plots with new data."""
        self.plot_with_design_point(
            self.axs[0, 0], pb_list, stability_list,
            self.texts["plot_bitumen"], self.texts["plot_stability"],
            obc, obc_values['Stabilite']
        )
        self.plot_with_design_point(
            self.axs[0, 1], pb_list, unit_weight_list,
            self.texts["plot_bitumen"], self.texts["plot_unit_weight"],
            obc, obc_values['UnitWeight']
        )
        self.plot_with_design_point(
            self.axs[1, 0], pb_list, va_list,
            self.texts["plot_bitumen"], self.texts["plot_va"],
            obc, obc_values['Va']
        )
        self.plot_with_design_point(
            self.axs[1, 1], pb_list, flow_list,
            self.texts["plot_bitumen"], self.texts["plot_flow"],
            obc, obc_values['Akma']
        )
        self.plot_with_design_point(
            self.axs[2, 0], pb_list, vma_list,
            self.texts["plot_bitumen"], self.texts["plot_vma"],
            obc, obc_values['VMA']
        )
        self.plot_with_design_point(
            self.axs[2, 1], pb_list, vfa_list,
            self.texts["plot_bitumen"], self.texts["plot_vfa"],
            obc, obc_values['VFA']
        )
        self.canvas.draw()

    def _display_results(self, obc, obc_values, specs, target_air_voids):
        """Format and display the final results in the text area."""
        lang_key = 'tr' if self.current_lang == 'TR' else 'en'
        selected_design_key = self.get_selected_design_key()
        design_type_name = DESIGN_CRITERIA[selected_design_key][lang_key]

        header = (
            f"{self.texts['results_design_type']}: {design_type_name}\n"
            f"{self.texts['results_obi']}: {obc:.2f}%\n"
            f"{self.texts['results_based_on_va'].format(va=target_air_voids)}\n"
            f"---------------------------------------------------\n"
            f"{self.texts['results_prop_header']:<20} "
            f"{self.texts['results_value_header']:>10}   "
            f"{self.texts['results_spec_header']:<15} "
            f"{self.texts['results_status']}\n"
            f"---------------------------------------------------\n"
        )

        spec_map = {
            'Va': f"{specs.get('Va', ('', ''))[0]}-{specs.get('Va', ('', ''))[1]}",
            'Stabilite': f">{specs.get('Stabilite', ('', ''))[0]}",
            'Akma': f"{specs.get('Akma', ('', ''))[0]}-{specs.get('Akma', ('', ''))[1]}",
            'VMA': f">{specs.get('VMA', ('', ''))[0]}",
            'VFA': f"{specs.get('VFA', ('', ''))[0]}-{specs.get('VFA', ('', ''))[1]}"
        }

        result_lines = []
        for key, value in obc_values.items():
            if key == 'UnitWeight':
                continue
            label = self.texts['prop_names'][key]
            spec_str = spec_map.get(key, '')
            status = self._check_spec(value, specs[key][0], specs[key][1])
            result_lines.append(
                f"{label:<20} {value:>9.2f}   {spec_str:<15} {status}"
            )

        self.results_text.config(state="normal")
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert("1.0", header + "\n".join(result_lines))
        self.results_text.config(state="disabled")

    def plot_with_design_point(self, ax, x, y, xlabel, ylabel, obi,
                             y_at_obi, poly_degree=2):
        """Plot the graph with a design point and fitted curve."""
        ax.clear()
        ax.plot(x, y, 'o', color='black')
        poly = None
        try:
            coeffs = np.polyfit(x, y, poly_degree)
            poly = np.poly1d(coeffs)
            x_smooth = np.linspace(min(x), max(x), 200)
            y_smooth = poly(x_smooth)
            ax.plot(x_smooth, y_smooth, '-', color='black')
        except (np.linalg.LinAlgError, TypeError):
            ax.plot(x, y, '-', color='gray')
        if obi is not None and y_at_obi is not None:
            ax.axvline(x=obi, color='g', linestyle='--', linewidth=1)
            ax.axhline(y=y_at_obi, color='g', linestyle='--', linewidth=1)
            ax.plot(obi, y_at_obi, 'r*', markersize=12)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.6)
        return poly


if __name__ == "__main__":
    app = MarshallApp()
    app.mainloop()