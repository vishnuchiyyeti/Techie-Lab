import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors
import py3Dmol
import streamlit.components.v1 as components

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch


st.set_page_config(
    page_title="Cheminformatics Virtual Laboratory",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { --navy:#17365D; --blue:#2F5597; --teal:#13A89E; }
.block-container {padding-top: 1.7rem; padding-bottom: 2.5rem; max-width: 1450px;}
.hero {
    padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1rem;
    background: linear-gradient(120deg, #102A43 0%, #17365D 55%, #167D8D 100%);
    color: white;
}
.hero h1 {font-size: 2.25rem; margin: 0 0 .35rem 0; color:white;}
.hero p {font-size: 1.02rem; margin: 0; color:#E5F4F7;}
.section-title {font-size:1.35rem; font-weight:700; color:#17365D; margin:.5rem 0;}
.metric-card {padding: .8rem 1rem; border:1px solid #D9E2EC; border-radius:12px; background:#F7FAFC;}
div[data-testid="stMetric"] {background:#F7FAFC; border:1px solid #D9E2EC; padding:12px 14px; border-radius:12px;}
.small-note {font-size:.88rem; color:#52606D;}
</style>
""", unsafe_allow_html=True)

EXAMPLE_MOLECULES = {
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Phenol": "Oc1ccccc1",
    "Acetic acid": "CC(=O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Glucose": "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O",
}

if "active_smiles" not in st.session_state:
    st.session_state.active_smiles = EXAMPLE_MOLECULES["Ethanol"]
if "selected_example" not in st.session_state:
    st.session_state.selected_example = "Ethanol"


def load_example():
    st.session_state.active_smiles = EXAMPLE_MOLECULES[st.session_state.selected_example]


def get_molecule(smiles):
    if not smiles or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol
    except Exception:
        return None


def calculate_bond_information(mol):
    # Sigma bonds include all atom-to-atom bonds after explicit H atoms are added.
    mol_h = Chem.AddHs(Chem.Mol(mol))
    sigma_bonds = mol_h.GetNumBonds()

    # Convert aromatic bonds to a Kekule representation to count pi components.
    kekule = Chem.Mol(mol)
    try:
        Chem.Kekulize(kekule, clearAromaticFlags=True)
    except Exception:
        pass

    pi_bonds = 0.0
    for bond in kekule.GetBonds():
        bt = bond.GetBondType()
        if bt == Chem.BondType.DOUBLE:
            pi_bonds += 1
        elif bt == Chem.BondType.TRIPLE:
            pi_bonds += 2
        elif bt == Chem.BondType.QUADRUPLE:
            pi_bonds += 3
        elif bond.GetIsAromatic():
            pi_bonds += 0.5

    return {
        "Heavy Atom Bonds": mol.GetNumBonds(),
        "Sigma (σ) Bonds": sigma_bonds,
        "Pi (π) Bonds": pi_bonds,
        "Total Bonds (σ + π)": sigma_bonds + pi_bonds,
        "Total Atoms Including H": mol_h.GetNumAtoms(),
    }


def calculate_properties(mol):
    bonds = calculate_bond_information(mol)
    return {
        "Molecular Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular Weight": round(Descriptors.MolWt(mol), 3),
        "Exact Molecular Weight": round(Descriptors.ExactMolWt(mol), 4),
        "LogP": round(Crippen.MolLogP(mol), 3),
        "TPSA (Å²)": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "Hydrogen Bond Donors": Lipinski.NumHDonors(mol),
        "Hydrogen Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable Bonds": Lipinski.NumRotatableBonds(mol),
        "Ring Count": Lipinski.RingCount(mol),
        "Aromatic Rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "Aliphatic Rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "Number of Atoms (heavy)": mol.GetNumHeavyAtoms(),
        "Heavy Atom Bonds": bonds["Heavy Atom Bonds"],
        "Sigma (σ) Bonds": bonds["Sigma (σ) Bonds"],
        "Pi (π) Bonds": bonds["Pi (π) Bonds"],
        "Total Bonds (σ + π)": bonds["Total Bonds (σ + π)"],
        "Fraction Csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "Molar Refractivity": round(Crippen.MolMR(mol), 3),
    }


def generate_2d_image(mol):
    return Draw.MolToImage(mol, size=(800, 550), kekulize=True)


def generate_3d_molecule(mol):
    try:
        mol3d = Chem.AddHs(Chem.Mol(mol))
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        status = AllChem.EmbedMolecule(mol3d, params)
        if status != 0:
            return None
        try:
            result = AllChem.MMFFOptimizeMolecule(mol3d, maxIters=300)
            if result < 0:
                AllChem.UFFOptimizeMolecule(mol3d, maxIters=300)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol3d, maxIters=300)
            except Exception:
                pass
        return mol3d
    except Exception:
        return None


def display_3d_structure(mol3d, style, show_surface=False):
    mol_block = Chem.MolToMolBlock(mol3d)
    view = py3Dmol.view(width=900, height=560)
    view.addModel(mol_block, "mol")
    if style == "Ball and Stick":
        view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.28}})
    elif style == "Stick":
        view.setStyle({"stick": {"radius": 0.18}})
    elif style == "Space Filling":
        view.setStyle({"sphere": {"scale": 0.85}})
    elif style == "Wireframe":
        view.setStyle({"line": {"linewidth": 2}})
    else:
        view.setStyle({"stick": {}, "sphere": {"scale": 0.20}})
    if show_surface:
        view.addSurface(py3Dmol.VDW, {"opacity": 0.55}, {"*": {}})
    view.setBackgroundColor("#fbfdff")
    view.zoomTo()
    components.html(view._make_html(), height=590, scrolling=False)


def generate_pdf(input_smiles, mol, properties):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=42, rightMargin=42,
        topMargin=42, bottomMargin=42
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "LabTitle", parent=styles["Title"], fontSize=19,
        textColor=HexColor("#17365D"), alignment=TA_CENTER, spaceAfter=12
    )
    heading_style = ParagraphStyle(
        "LabHeading", parent=styles["Heading2"], fontSize=13,
        textColor=HexColor("#2F5597"), spaceBefore=10, spaceAfter=7
    )
    body_style = ParagraphStyle(
        "LabBody", parent=styles["BodyText"], fontSize=9, leading=13
    )
    story = [
        Paragraph("CHEMINFORMATICS VIRTUAL LABORATORY", title_style),
        Paragraph("Molecular Properties Report", styles["Heading2"]),
        Spacer(1, 0.12 * inch),
        Paragraph("<b>Input SMILES:</b> " + input_smiles.replace("&", "&amp;").replace("<", "&lt;"), body_style),
        Paragraph("<b>Canonical SMILES:</b> " + Chem.MolToSmiles(mol), body_style),
        Paragraph("<b>Molecular Formula:</b> " + properties["Molecular Formula"], body_style),
        Spacer(1, 0.2 * inch),
        Paragraph("Calculated Molecular Properties", heading_style),
    ]
    data = [["Property", "Value"]] + [[str(k), str(v)] for k, v in properties.items()]
    table = Table(data, colWidths=[3.5 * inch, 2.5 * inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HexColor("#F3F7FB")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([
        table, Spacer(1, 0.2 * inch),
        Paragraph("Interpretation and limitations", heading_style),
        Paragraph(
            "Properties in this report are computed from the supplied molecular "
            "structure using RDKit. They are cheminformatics descriptors, not "
            "experimental measurements or medical advice. Bond counts are based "
            "on the toolkit's molecular representation; aromaticity and resonance "
            "can make simplified sigma/pi counts convention-dependent.",
            body_style
        )
    ])
    doc.build(story)
    result = buffer.getvalue()
    buffer.close()
    return result


def interpretation(name, value):
    notes = {
        "Molecular Formula": "Elemental composition of the molecule.",
        "Molecular Weight": "Average molecular mass, in g/mol.",
        "Exact Molecular Weight": "Monoisotopic mass, useful for mass-spectrometry comparisons.",
        "LogP": "Estimated octanol/water partition coefficient; higher values generally indicate greater lipophilicity.",
        "TPSA (Å²)": "Topological polar surface area, often used as a descriptor in permeability analysis.",
        "Hydrogen Bond Donors": "Estimated number of hydrogen-bond donor groups.",
        "Hydrogen Bond Acceptors": "Estimated number of hydrogen-bond acceptor atoms.",
        "Rotatable Bonds": "A descriptor related to molecular flexibility.",
        "Ring Count": "Number of rings identified by RDKit's ring-count descriptor.",
        "Aromatic Rings": "Number of aromatic rings.",
        "Aliphatic Rings": "Number of non-aromatic rings.",
        "Number of Atoms (heavy)": "Number of non-hydrogen atoms.",
        "Heavy Atom Bonds": "Number of bonds in the input heavy-atom molecular graph.",
        "Sigma (σ) Bonds": "Count of atom-to-atom bonds after adding explicit hydrogens.",
        "Pi (π) Bonds": "Approximate π-bond count from a Kekulé representation; resonance makes this convention-dependent.",
        "Total Bonds (σ + π)": "Approximate sigma-plus-pi count; aromatic systems require care in interpretation.",
        "Fraction Csp3": "Fraction of carbon atoms classified as sp³-hybridized.",
        "Molar Refractivity": "A calculated descriptor related to molecular volume and polarizability.",
    }
    return notes.get(name, "Calculated molecular descriptor.")


st.markdown("""
<div class="hero">
  <h1>🧪 Cheminformatics Virtual Laboratory</h1>
  <p>Explore molecular structures, calculate physicochemical descriptors, and export a clean lab report.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Molecule input")
    st.selectbox("Load an example molecule", list(EXAMPLE_MOLECULES.keys()), key="selected_example", on_change=load_example)
    smiles = st.text_area(
        "SMILES notation",
        key="active_smiles",
        height=100,
        help="Paste a valid SMILES string to analyse a molecule."
    )
    analyse = st.button("🔬 Analyse molecule", type="primary", use_container_width=True)
    st.caption("Examples are provided for quick exploration. Your input is processed in this app session.")

if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = True
if analyse:
    st.session_state.run_analysis = True

mol = get_molecule(smiles) if st.session_state.run_analysis else None

if not smiles.strip():
    st.info("Enter a SMILES string or choose an example molecule from the sidebar.")
    st.stop()
if mol is None:
    st.error("This SMILES string could not be parsed. Check the notation or select an example molecule.")
    st.stop()

properties = calculate_properties(mol)
canonical = Chem.MolToSmiles(mol)

top1, top2, top3, top4 = st.columns(4)
top1.metric("Molecular formula", properties["Molecular Formula"])
top2.metric("Molecular weight", f'{properties["Molecular Weight"]:.3f} g/mol')
top3.metric("Heavy atoms", str(properties["Number of Atoms (heavy)"]))
top4.metric("Rotatable bonds", str(properties["Rotatable Bonds"]))

tab_structure, tab_properties, tab_3d, tab_report = st.tabs([
    "🧬 2D Structure", "📊 Properties", "🌀 3D Viewer", "📄 Export report"
])

with tab_structure:
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("2D molecular structure")
        st.image(generate_2d_image(mol), use_container_width=True)
    with right:
        st.subheader("Molecular identifiers")
        st.text_input("Canonical SMILES", value=canonical, disabled=True)
        st.text_input("Molecular formula", value=properties["Molecular Formula"], disabled=True)
        st.markdown("**Structure summary**")
        st.write(f'- **Heavy atoms:** {mol.GetNumHeavyAtoms()}')
        st.write(f'- **Heavy-atom bonds:** {mol.GetNumBonds()}')
        st.write(f'- **Rings:** {properties["Ring Count"]}')
        st.caption("2D depiction is generated by RDKit from the parsed molecular graph.")

with tab_properties:
    st.subheader("Calculated molecular descriptors")
    prop_df = pd.DataFrame(
        [{"Property": k, "Value": v, "Interpretation": interpretation(k, v)} for k, v in properties.items()]
    )
    st.dataframe(prop_df, use_container_width=True, hide_index=True)
    csv_data = prop_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download properties as CSV", csv_data, "molecular_properties.csv", "text/csv")
    st.markdown("#### Quick descriptor overview")
    numeric_props = {k: v for k, v in properties.items() if isinstance(v, (int, float)) and k not in ("Pi (π) Bonds",)}
    fig, ax = plt.subplots(figsize=(9, 4))
    names = list(numeric_props.keys())
    vals = list(numeric_props.values())
    ax.bar(range(len(names)), vals)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=65, ha="right", fontsize=8)
    ax.set_ylabel("Descriptor value")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab_3d:
    st.subheader("Interactive 3D molecular viewer")
    col1, col2 = st.columns([1, 1])
    with col1:
        view_style = st.selectbox("Representation", ["Ball and Stick", "Stick", "Space Filling", "Wireframe", "Cartoon"])
    with col2:
        show_surface = st.checkbox("Show van der Waals surface", value=False)
    with st.spinner("Generating a 3D conformer..."):
        mol3d = generate_3d_molecule(mol)
    if mol3d is None:
        st.warning("A 3D conformer could not be generated for this structure.")
    else:
        display_3d_structure(mol3d, view_style, show_surface)
        st.caption("The 3D conformer is computationally generated and energy-minimised where supported. It is one plausible conformer, not necessarily the experimentally observed geometry.")

with tab_report:
    st.subheader("Export your analysis")
    st.write("Create a PDF report containing the input structure identifier and calculated descriptors.")
    st.info("The report includes calculated values and a note about interpretation limits.")
    pdf_bytes = generate_pdf(smiles.strip(), mol, properties)
    st.download_button(
        "📄 Download molecular properties report (PDF)",
        data=pdf_bytes,
        file_name="molecular_properties_report.pdf",
        mime="application/pdf",
        type="primary"
    )
    st.download_button(
        "⬇️ Download raw properties (CSV)",
        data=prop_df.to_csv(index=False).encode("utf-8"),
        file_name="molecular_properties.csv",
        mime="text/csv"
    )

st.divider()
st.markdown(
    '<p class="small-note">Educational tool only. Calculated descriptors depend on the structure and toolkit conventions; verify important results against trusted scientific references.</p>',
    unsafe_allow_html=True
)
