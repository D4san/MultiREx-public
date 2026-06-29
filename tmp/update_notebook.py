import json
import os

nb_path = r"c:\Proyetos\Repos\MultiREx-public\examples\multirex-chemical_equilibrium.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Insert a Markdown cell about optional dependencies near the top
md_install = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Important: Optional Dependencies\n",
        "To run this notebook fully, you need the optional `ggchem` and `cia` dependencies. You can install them via:\n",
        "```bash\n",
        "pip install \"multirex[ggchem,cia]\"\n",
        "```\n"
    ]
}
nb["cells"].insert(1, md_install) # right after the title

# Append CIA tutorial at the bottom
cells_to_append = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Including Collision-Induced Absorption (CIA)\n",
            "\n",
            "In H2/He dominated atmospheres, Collision-Induced Absorption (CIA) can be a significant source of opacity, especially in the infrared. MultiREx allows you to include CIA easily by specifying the `cia` parameter in the `Atmosphere` object.\n",
            "\n",
            "First, let's download and configure the CIA data cache using `Util.get_CIAs()`:"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import multirex.utils as Util\n",
            "\n",
            "# This will download the CIA cross sections for H2-H2 and H2-He\n",
            "# and configure the TauREx cache automatically.\n",
            "Util.get_CIAs(pairs=['H2-H2', 'H2-He'], path='./cia_data')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "Now, let's recreate our planet's atmosphere adding the `cia` parameter, and generate a new transmission model."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create atmosphere with GGChem and CIA\n",
            "atm_cia = mrex.Atmosphere(\n",
            "    temperature=1500,\n",
            "    base_pressure=1e6,\n",
            "    top_pressure=1e-4,\n",
            "    chemistry_type='ggchem',\n",
            "    ggchem_params=ggchem_params,\n",
            "    cia=['H2-H2', 'H2-He']\n",
            ")\n",
            "\n",
            "planet_cia = mrex.Planet(radius=1.5, mass=1.0, atmosphere=atm_cia)\n",
            "system_cia = mrex.System(star=star, planet=planet_cia, sma=0.05)\n",
            "\n",
            "# Build the new transmission model\n",
            "system_cia.make_tm()\n",
            "\n",
            "# Generate the spectrum\n",
            "wns = mrex.Physics.wavenumber_grid(wl_min=0.3, wl_max=10, resolution=500)\n",
            "bin_wn_cia, bin_rprs_cia = system_cia.generate_spectrum(wns)\n",
            "\n",
            "# Plot and compare\n",
            "import matplotlib.pyplot as plt\n",
            "plt.figure(figsize=(10, 5))\n",
            "plt.plot(10000/bin_wn, bin_rprs, label='Without CIA', alpha=0.7)\n",
            "plt.plot(10000/bin_wn_cia, bin_rprs_cia, label='With CIA', alpha=0.7)\n",
            "plt.xscale('log')\n",
            "plt.xlabel('Wavelength [μm]')\n",
            "plt.ylabel('$(R_p/R_s)^2$')\n",
            "plt.legend()\n",
            "plt.show()"
        ]
    }
]

nb["cells"].extend(cells_to_append)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
