"""
MinPop: minimal-basis (MBS) Mulliken population analysis in PySCF.

Drivers (each also a command-line tool):
    minpop.minpop_rohf   RHF / ROHF reference
    minpop.minpop_uhf    UHF reference (with spin annihilation)
    minpop.minpop_rks    RKS / ROKS reference
    minpop.minpop_uks    UKS reference
    minpop.minpop_gvbpp  GVB-PP reference (keithgroup/gvbpp)
Modules:
    minpop.bond_symmetry                  sigma/pi/delta/phi decomposition of bond populations
    minpop.basis_sets.anorcc_basis_pyscf   ANO-RCC-MB / VDZP / VTZP / VQZP
    minpop.minpop_json_csv           JSON / CSV export of finished reports

The heavy modules are imported lazily; `import minpop` is cheap.
"""

__version__ = "2.0.0"
__all__ = ["run_rohf_from_xyz", "run_uhf_from_xyz", "run_rks_from_xyz",
           "run_uks_from_xyz", "run_gvbpp_from_xyz", "bond_components",
           "class_matrices", "__version__"]


def __getattr__(name):
    import importlib
    table = {
        "run_rohf_from_xyz": ("minpop.minpop_rohf", "run_rohf_from_xyz"),
        "run_uhf_from_xyz": ("minpop.minpop_uhf", "run_uhf_from_xyz"),
        "run_rks_from_xyz": ("minpop.minpop_rks", "run_rks_from_xyz"),
        "run_uks_from_xyz": ("minpop.minpop_uks", "run_uks_from_xyz"),
        "run_gvbpp_from_xyz": ("minpop.minpop_gvbpp", "run_gvbpp_from_xyz"),
        "bond_components": ("minpop.bond_symmetry", "bond_components"),
        "class_matrices": ("minpop.bond_symmetry", "class_matrices"),
    }
    if name in table:
        mod, attr = table[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(name)
