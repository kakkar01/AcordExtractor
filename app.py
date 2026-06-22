import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import fitz
import streamlit as st

import config

PROJECT_ROOT = Path(__file__).resolve().parent


def _summarize_status(status_value: Any) -> str:
    if not status_value:
        return "—"
    if isinstance(status_value, list):
        first = status_value[0] if status_value else {}
        if isinstance(first, dict):
            selected = [name for name, is_selected in first.items() if is_selected]
            return ", ".join(selected) if selected else "None"
    return str(status_value)


st.set_page_config(
    page_title="ACORD Extractor Review",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Welcome to the World of Insurance")
st.markdown(
    "### Your ACORD review companion for fast, friendly extraction. "
    "Upload a file and we’ll guide you through the results with clear, easy-to-read summaries."
)
st.markdown(
    "<div style='padding:12px; border:1px solid #e2e2e2; border-radius:12px; background:#f8f9fa;'>"
    "<strong>Ready to explore your policy data?</strong> "
    "We’ll extract your transaction status, locations, coverage lines, and more." 
    "</div>",
    unsafe_allow_html=True,
)
st.caption("A shareable Streamlit app for reviewing uploaded ACORD forms and extracted data.")

with st.sidebar:
    st.markdown("## Ready to start? 🚀")
    st.markdown(
        "Upload an ACORD PDF or image and we’ll extract the key details for you. "
        "This process can take about 1 minute — please bear with me while I work through it."
    )
    st.info("Tip: For best results, upload a clean scan or PDF of your ACORD form.")
    st.write("---")
    st.header("Upload")
    uploaded_file = st.file_uploader(
        "Choose a PDF or image",
        type=["pdf", "png", "jpg", "jpeg", "tiff"],
    )
    form_name = st.selectbox(
        "Please specify Form type",
        ["", "ACORD 125", "ACORD 126", "ACORD 131", "ACORD 139", "ACORD 140"],
        index=0,
    )
    run_button = st.button("Run extraction", type="primary")

if uploaded_file is None:
    st.info("Upload a PDF or image from the sidebar to begin.")
    st.stop()

if run_button:
    with st.spinner("Processing your ACORD file — this may take around 1 minute. Please bear with me..."):
        suffix = Path(uploaded_file.name).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = Path(tmp.name)

        try:
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "main.py"),
                "--backend",
                config.INFERENCE_BACKEND,
                "extract",
                str(temp_path),
                "--no-save",
                "--markdown",
            ]
            if form_name:
                cmd.extend(["--form", form_name])

            completed = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=1800,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    (completed.stderr or completed.stdout).strip()
                    or "Extraction failed without output."
                )

            result = json.loads(completed.stdout)
            st.session_state["result"] = result
            st.session_state["source_name"] = uploaded_file.name
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

if "result" not in st.session_state:
    st.stop()

result = st.session_state["result"]
source_name = st.session_state.get("source_name", uploaded_file.name)

# Display extraction accuracy
extraction_accuracy = result.get("_extraction_accuracy", 90)
accuracy_color = "🟢" if extraction_accuracy >= 95 else "🟡" if extraction_accuracy >= 90 else "🟠"
st.markdown(
    f"<div style='padding:12px; background:#f0f8ff; border-left:4px solid #4CAF50; border-radius:4px; margin-bottom:20px;'>"
    f"<strong>{accuracy_color} Extraction Confidence:</strong> <span style='font-size:1.2em; font-weight:bold; color:#1976D2;'>{extraction_accuracy}%</span>"
    f"</div>",
    unsafe_allow_html=True,
)

def _format_address(address: Any) -> str:
    if not isinstance(address, dict):
        return str(address) if address else ""
    street = (
        address.get("street")
        or address.get("address_line")
        or address.get("address_line1")
        or address.get("street_address")
    )
    parts = [
        street,
        address.get("city"),
        address.get("state"),
        address.get("county"),
        address.get("zip"),
    ]
    return ", ".join(str(part) for part in parts if part)


def _display(label: str, value: Any) -> None:
    if value is None or value == "":
        return
    st.markdown(f"- **{label}:** {value}")


def _display_selected_company_type(options: list[Any]) -> None:
    if not isinstance(options, list) or not options:
        return
    selected = [item.get("company_type") for item in options if isinstance(item, dict) and item.get("selected")]
    if selected:
        st.markdown("- **Company Type:**")
        for value in selected:
            st.markdown(f"  - {value}")


def _value_from_sources(keys: list[str], sources: list[Any]) -> Any:
    for key in keys:
        for source in sources:
            if isinstance(source, dict) and key in source:
                value = source.get(key)
                if value is not None and value != "":
                    return value
    return None


def _display_from_sources(label: str, keys: list[str], sources: list[Any]) -> None:
    value = _value_from_sources(keys, sources)
    if value is not None and value != "":
        _display(label, value)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


EXPECTED_LOBS = [
    "Boiler & Machinery",
    "Business Auto",
    "Business Owners",
    "Commercial General Liability",
    "Commercial Inland Marine",
    "Commercial Property",
    "Crime",
    "Cyber and Privacy",
    "Fiduciary Liability",
    "Garage and Dealers",
    "Liquor Liability",
    "Motor Carrier",
    "Truckers",
    "Umbrella",
    "Yacht",
]


def _find_lob_entry(lines_of_business: list[Any], expected_label: str) -> dict[str, Any] | None:
    expected_norm = _normalize_text(expected_label)
    for item in lines_of_business:
        if not isinstance(item, dict):
            continue
        lob_name = item.get("lob") or item.get("name") or ""
        lob_norm = _normalize_text(lob_name)
        if not lob_norm:
            continue
        if expected_norm == lob_norm or expected_norm in lob_norm or lob_norm in expected_norm:
            return item
    return None


def _display_selected_lobs(lines_of_business: list[Any]) -> None:
    selected = [item for item in lines_of_business if isinstance(item, dict) and item.get("selected") is True]
    st.markdown("- **Selected Lines of Business:**")
    if not selected:
        st.markdown("  - None")
        return
    for item in selected:
        lob_name = item.get("lob") or item.get("name") or "Unknown"
        premium = item.get("premium") or item.get("amount") or item.get("limit")
        if premium:
            st.markdown(f"  - {lob_name} — {premium}")
        else:
            st.markdown(f"  - {lob_name}")


def _display_selected_attachments(attachments: list[Any]) -> None:
    selected = [item for item in attachments if isinstance(item, dict) and item.get("selected") is True]
    st.markdown("- **Selected Attachments:**")
    if not selected:
        st.markdown("  - None")
        return
    for item in selected:
        attachment_type = item.get("attachment_type") or item.get("name") or item.get("label") or "Attachment"
        st.markdown(f"  - {attachment_type}")


def _render_record(title: str, record: Any, field_map: list[tuple[str, str]]) -> None:
    if not isinstance(record, dict):
        return
    st.markdown(f"- **{title}:**")
    for label, key in field_map:
        value = record.get(key)
        if key == "mailing_address":
            value = _format_address(value)
        _display(label, value)


def _render_record_list(title: str, records: list[Any], field_map: list[tuple[str, str]]) -> None:
    if not isinstance(records, list) or not records:
        st.markdown(f"- **{title}:** None")
        return
    st.markdown(f"- **{title}:**")
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            st.markdown(f"  - Record {index}: {record}")
            continue
        st.markdown(f"  - Record {index}:")
        for label, key in field_map:
            if key == "full_address":
                value = _format_address(record)
            else:
                value = record.get(key)
            if value:
                _display(label, value)


# Detect form type based on available fields in result
def _detect_form_type(result: dict) -> str:
    """Detect ACORD form type from extracted result."""
    if "acord140Policy" in result or "acord140PremisesInformation" in result:
        return "ACORD 140"
    elif "acord126" in result or "acord126Hazards" in result:
        return "ACORD 126"
    elif "acord131Policy" in result or "acord131Location" in result:
        return "ACORD 131"
    elif "acord139" in result or "acord139Location" in result:
        return "ACORD 139"
    return "ACORD 125"


form_type = _detect_form_type(result) if isinstance(result, dict) else "ACORD 125"

# Extract form-specific fields
if form_type == "ACORD 140":
    acord140_policy = result.get("acord140Policy", {}) if isinstance(result, dict) else {}
    acord140_premises = result.get("acord140PremisesInformation", []) if isinstance(result, dict) else []
    acord140_blanket = result.get("acord140BlanketSummary", {}) if isinstance(result, dict) else {}
elif form_type == "ACORD 126":
    acord126_policy = result.get("acord126", {}) if isinstance(result, dict) else {}
    acord126_hazards = result.get("acord126Hazards", []) if isinstance(result, dict) else []
elif form_type == "ACORD 131":
    acord131_policy = result.get("acord131Policy", {}) if isinstance(result, dict) else {}
    acord131_locations = result.get("acord131Location", []) if isinstance(result, dict) else []
    acord131_underlying = result.get("acord131UnderlyingInsurance", []) if isinstance(result, dict) else []
    acord131_extra = result.get("extraUnderlyingInsurance", []) if isinstance(result, dict) else []
elif form_type == "ACORD 139":
    acord139_policy = result.get("acord139", {}) if isinstance(result, dict) else {}
    acord139_locations = result.get("acord139Location", []) if isinstance(result, dict) else []
else:
    # ACORD 125
    policy_header = result.get("policy_header", {}) if isinstance(result, dict) else {}
    policy_information = result.get("policy_information", {}) if isinstance(result, dict) else {}
    applicant = result.get("applicant", {}) if isinstance(result, dict) else {}
    insured = result.get("insured", {}) if isinstance(result, dict) else {}
    location_records = result.get("locations") or result.get("premises") or []
    prior_carrier = result.get("prior_carrier_information") or result.get("prior_carrier_table") or result.get("prior_carrier")
    loss_history = result.get("loss_history") or result.get("prior_losses") or []
    lines_of_business = result.get("lines_of_business", []) if isinstance(result, dict) else []
    attachments = result.get("attachments", []) if isinstance(result, dict) else []

col1, col2 = st.columns([1, 1])
with col1:
    # Display accuracy metric at the top
    extraction_accuracy = result.get("_extraction_accuracy", 90)
    accuracy_color_hex = "#4CAF50" if extraction_accuracy >= 95 else "#FF9800" if extraction_accuracy >= 90 else "#F44336"
    st.metric(
        label="Extraction Confidence",
        value=f"{extraction_accuracy}%",
        delta=None,
        delta_color="off"
    )
    st.divider()
    
    if form_type == "ACORD 140":
        st.subheader("ACORD 140 Summary")
        st.markdown("Review the extracted ACORD 140 sections below.")

        with st.expander("1. Policy Information (acord140Policy)", expanded=True):
            _display("Agency", acord140_policy.get("agency"))
            _display("Carrier", acord140_policy.get("carrier"))
            _display("NAIC Code", acord140_policy.get("naicCode"))
            _display("Effective Date", acord140_policy.get("effectiveDate"))
            _display("Applicant/First Named Insured", acord140_policy.get("applicantFirstNamedInsured"))
            _display("Construction Type (Primary Premises)", acord140_policy.get("constructionTypePremisesInformation"))
            _display("Fire District (Primary)", acord140_policy.get("fireDistrictPremisesInformation"))
            _display("Distance to Fire Station (Primary)", acord140_policy.get("distanceToFireStationPremisesInformation"))
            _display("Distance to Hydrant (Primary)", acord140_policy.get("distanceToHydrantPremisesInformation"))
            _display("Protection Class (Primary)", acord140_policy.get("protClassPremisesInformation"))
            _display("Number of Stories (Primary)", acord140_policy.get("numberStoriesPremisesInformation"))
            _display("Year Built (Primary)", acord140_policy.get("yearBuiltPremisesInformation"))
            _display("Total Area (Primary)", acord140_policy.get("totalAreaPremisesInformation"))
            _display("Roof Type (Primary)", acord140_policy.get("roofTypePremisesInformation"))
            _display("Roof Year (Primary)", acord140_policy.get("roofingyearPremisesInformation"))
            _display("Heating Year (Primary)", acord140_policy.get("heatingYearPremisesInformation"))
            _display("Wiring Year (Primary)", acord140_policy.get("wiringYearPremisesInformation"))
            _display("Plumbing Year (Primary)", acord140_policy.get("plumbingYearPremisesInformation"))
            _display("Burglar Alarm Type (Primary)", acord140_policy.get("burglarAlarmTypePremisesInformation"))
            _display("Fire Protection (Primary)", acord140_policy.get("premisesFireProtectionPremisesInformation"))
            _display("Sprinkler Coverage % (Primary)", acord140_policy.get("percentSprinklerPremisesInformation"))

        with st.expander("2. Premises Information (acord140PremisesInformation)", expanded=True):
            if not acord140_premises:
                st.markdown("- No premises information found.")
            elif isinstance(acord140_premises, dict):
                # Display dict row-wise
                st.markdown("- **Premises Information:**")
                for key, value in acord140_premises.items():
                    if value is None or value == "":
                        continue
                    if isinstance(value, list):
                        value_str = ", ".join(str(v) for v in value if v)
                    else:
                        value_str = str(value)
                    if value_str:
                        # Convert snake_case to Title Case
                        label = key.replace("_", " ").title()
                        st.markdown(f"  - **{label}:** {value_str}")
            elif isinstance(acord140_premises, list):
                for index, premise in enumerate(acord140_premises, start=1):
                    if not isinstance(premise, dict):
                        st.markdown(f"- Premises {index}: {premise}")
                        continue
                    st.markdown(f"- **Premises {index}**")
                    _display("Premises Number", premise.get("premises_number"))
                    _display("Building Number", premise.get("building_number"))
                    _display("Street Address", premise.get("street_address"))
                    _display("Building Description", premise.get("bldg_desc"))
                    _display("Subject of Insurance", premise.get("subject_of_insurance"))
                    _display("Amount", premise.get("amount"))
                    _display("COINS %", premise.get("coins_percent"))
                    _display("Valuation", premise.get("valuation"))
                    _display("Causes of Loss", premise.get("causes_of_losses"))
                    _display("Inflation Guard %", premise.get("inflation_guard_percent"))
                    _display("Deductible", premise.get("ded"))
                    _display("Deductible Type", premise.get("ded_type"))
                    _display("Blanket Number", premise.get("blkt_number"))
                    _display("Forms and Conditions", premise.get("forms_and_conditions_to_apply"))
            else:
                st.markdown(f"- {acord140_premises}")

        with st.expander("3. Blanket Summary (acord140BlanketSummary)", expanded=True):
            if acord140_blanket:
                _display("Blanket Number", acord140_blanket.get("blkt_number"))
                _display("Amount", acord140_blanket.get("amount"))
                _display("Blanket Type", acord140_blanket.get("blkt_type"))
            else:
                st.markdown("- No blanket summary found.")

    elif form_type == "ACORD 126":
        st.subheader("ACORD 126 Summary")
        st.markdown("Review the extracted ACORD 126 General Liability sections below.")

        with st.expander("1. Policy Information (acord126)", expanded=True):
            _display("Agency", acord126_policy.get("agency"))
            _display("Carrier", acord126_policy.get("carrier"))
            _display("NAIC Code", acord126_policy.get("naicCode"))
            _display("Policy Number", acord126_policy.get("policyNumber"))
            _display("Effective Date", acord126_policy.get("effectiveDate"))
            _display("Applicant/First Named Insured", acord126_policy.get("applicantFirstNamedInsured"))
            _display("Coverage Type", acord126_policy.get("coveragesClaimsMadeOrOccurrence_selected"))
            _display("Products Premium", acord126_policy.get("ProductsPremium"))
            _display("Other Premium", acord126_policy.get("OtherPremium"))
            _display("Premises/Operations Premium", acord126_policy.get("PremisesOperationsPremium"))
            _display("Total Premium", acord126_policy.get("TotalPremium"))
            _display("Limits Apply Per", acord126_policy.get("limitsappliesper_selected"))
            _display("General Aggregate Limit", acord126_policy.get("GeneralAggregateLimitType"))
            _display("Products & Completed Operations Aggregate Limit", acord126_policy.get("ProductsCompletedOperationsAggregateLimitType"))
            _display("Personal & Advertising Injury Limit", acord126_policy.get("PersonalAdvertisingInjuryLimitType"))
            _display("Each Occurrence Limit", acord126_policy.get("EachOccurenceLimitType"))
            _display("Damage to Rented Premises Limit", acord126_policy.get("DamageToRentedPremisesLimitType"))
            _display("Medical Expense Limit", acord126_policy.get("MedicalExpenseLimitType"))
            _display("Employee Benefits Limit", acord126_policy.get("EmployeeBenefitsLimitType"))
            _display("Deductibles", acord126_policy.get("deductibletypeandamount"))
            _display("Proposed Retroactive Date", acord126_policy.get("proposedRetroactiveDate"))
            _display("Tail Coverage Purchased", acord126_policy.get("wasTailCoveragePurchased"))
            _display("License", acord126_policy.get("license"))

        with st.expander("2. Schedule of Hazards (acord126Hazards)", expanded=True):
            if not acord126_hazards:
                st.markdown("- No hazards found.")
            else:
                for index, hazard in enumerate(acord126_hazards, start=1):
                    if not isinstance(hazard, dict):
                        st.markdown(f"- Hazard {index}: {hazard}")
                        continue
                    st.markdown(f"- **Hazard {index}**")
                    _display("Location Number", hazard.get("location_number"))
                    _display("Hazard Number", hazard.get("hazard_number"))
                    _display("Classification", hazard.get("classification_class"))
                    _display("Class Code", hazard.get("class_code"))
                    _display("Premium Basis", hazard.get("premium_basis"))
                    _display("Exposure", hazard.get("exposure"))
                    _display("Rate Prem/Ops", hazard.get("rate_prem_ops"))
                    _display("Rate Products", hazard.get("rate_products"))

    elif form_type == "ACORD 131":
        st.subheader("ACORD 131 Summary")
        st.markdown("Review the extracted ACORD 131 Umbrella/Excess Liability sections below.")

        with st.expander("1. Policy Information (acord131Policy)", expanded=True):
            _display("License", acord131_policy.get("license"))
            _display("NAIC Code", acord131_policy.get("naic_code"))
            _display("Agency", acord131_policy.get("agency"))
            _display("Policy Number", acord131_policy.get("policy_number"))
            _display("Named Insured", acord131_policy.get("named_insured"))
            _display("Carrier", acord131_policy.get("carrier"))
            _display("Effective Date", acord131_policy.get("effective_date"))
            _display("Transaction Type", acord131_policy.get("transaction_type"))
            _display("Proposed Retroactive Date", acord131_policy.get("retroactivedate_proposed"))
            _display("Current Retroactive Date", acord131_policy.get("retroactivedate_current"))
            _display("Aggregate Limit of Liability", acord131_policy.get("agg_limit_liability"))
            _display("Products & Completed Operations Limit", acord131_policy.get("prod_comp_ops_limit_liability"))
            _display("Each Occurrence Limit", acord131_policy.get("ea_occ_limit_liability"))
            _display("Retained Limit", acord131_policy.get("retained_limit"))
            _display("First Dollar Defense", acord131_policy.get("first_dollar_defense"))
            _display("Limit of Insurance", acord131_policy.get("limit_of_insurance"))

        with st.expander("2. Locations (acord131Location)", expanded=True):
            if not acord131_locations:
                st.markdown("- No location data found.")
            else:
                for index, location in enumerate(acord131_locations, start=1):
                    if not isinstance(location, dict):
                        st.markdown(f"- Location {index}: {location}")
                        continue
                    st.markdown(f"- **Location {index}**")
                    _display("Location Number", location.get("location_number"))
                    _display("Location Name", location.get("location_name"))
                    _display("Annual Payroll", location.get("annual_payroll"))
                    _display("Annual Gross Sales", location.get("annual_grossSales"))
                    _display("Foreign Gross Sales", location.get("foreign_grossSales"))
                    _display("Number of Employees", location.get("number_employees"))
                    _display("Name", location.get("name"))
                    _display("Description", location.get("description"))

        with st.expander("3. Underlying Insurance (acord131UnderlyingInsurance)", expanded=True):
            if not acord131_underlying:
                st.markdown("- No underlying insurance found.")
            else:
                for index, underlying in enumerate(acord131_underlying, start=1):
                    if not isinstance(underlying, dict):
                        st.markdown(f"- Underlying {index}: {underlying}")
                        continue
                    ins_type = underlying.get("type", f"Insurance {index}")
                    st.markdown(f"- **{ins_type} {index}**")
                    _display("Policy Number", underlying.get("CarrierPolicyNumber"))
                    _display("Effective Date", underlying.get("policy_eff_date"))
                    _display("Expiration Date", underlying.get("policy_exp_date"))
                    _display("Rating Mod", underlying.get("rating_mod"))
                    if ins_type == "AutomobileLiability":
                        _display("CSL Limit", underlying.get("cslea_acc__limit_amount"))
                        _display("BI Each Accident", underlying.get("biea_acc_limit_amount"))
                        _display("BI Each Person", underlying.get("biea_per_limit_amount"))
                        _display("PD Each Accident", underlying.get("pdea_acc_limit_amount"))
                        _display("Annual Renewal Premium", underlying.get("automobile_liability_annual_renewal_premium"))
                    elif ins_type == "GeneralLiability":
                        _display("Policy Type", underlying.get("generalliability_policy_type"))
                        _display("Each Occurrence Limit", underlying.get("each_occurrence_limit_amount"))
                        _display("General Aggregate Limit", underlying.get("general_aggr_limit_amount"))
                        _display("Personal & Adv Injury Limit", underlying.get("personal_and_adv_injury_limit_amount"))
                        _display("Damage to Rented Premises", underlying.get("damage_to_rented_premises_limit_amount"))
                        _display("Medical Expense Limit", underlying.get("medical_expense_limit_amount"))
                        _display("Products & Completed Ops Limit", underlying.get("prod_comp_op_aggregate_limit_amount"))
                        _display("Prem/Ops Premium", underlying.get("generalliability_prem_ops"))
                        _display("Products Premium", underlying.get("generalliability_products"))
                        _display("Other Premium", underlying.get("generalliability_other"))
                    elif ins_type == "EmployersLiability":
                        _display("Each Accident Limit", underlying.get("each_accident_limit_amount"))
                        _display("Disease per Employee Limit", underlying.get("disease_each_employee_limit_amount"))
                        _display("Disease Policy Limit", underlying.get("disease_policy_limit_limit_amount"))
                        _display("Annual Renewal Premium", underlying.get("employersliability_annual_renewal_premium"))

        with st.expander("4. Extra Underlying Insurance (extraUnderlyingInsurance)", expanded=True):
            if not acord131_extra:
                st.markdown("- No extra underlying insurance found.")
            else:
                for index, extra in enumerate(acord131_extra, start=1):
                    if not isinstance(extra, dict):
                        st.markdown(f"- Extra {index}: {extra}")
                        continue
                    st.markdown(f"- **Extra {index}**")
                    _display("Type", extra.get("type"))
                    _display("Policy Number", extra.get("CarrierPolicyNumber"))
                    _display("Effective Date", extra.get("policy_eff_date"))
                    _display("Expiration Date", extra.get("policy_exp_date"))
                    _display("Limit Type", extra.get("other_limit_type"))
                    _display("Limit Amount", extra.get("other_limit_amount"))
                    _display("Annual Renewal Premium", extra.get("other_annual_renewal_premium"))
                    _display("Rating Mod", extra.get("rating_mod"))

    elif form_type == "ACORD 139":
        st.subheader("ACORD 139 Summary")
        st.markdown("Review the extracted ACORD 139 Commercial Property sections below.")

        with st.expander("1. Policy Information (acord139)", expanded=True):
            _display("Named Insured", acord139_policy.get("named_insured"))
            _display("Agency Name & Address", acord139_policy.get("AgencyNameAddress"))
            _display("Carrier", acord139_policy.get("carrier"))
            _display("Effective Date", acord139_policy.get("effective_date"))
            _display("NAIC Code", acord139_policy.get("naic_code"))
            _display("Contact Name", acord139_policy.get("contact_name"))
            _display("Phone", acord139_policy.get("phone"))
            _display("Email", acord139_policy.get("email_address"))
            _display("Coinsurance %", acord139_policy.get("coins_percent"))
            _display("Applicable Cause of Loss", acord139_policy.get("Applicable_Cause_Of_Loss"))
            _display("Date of Application", acord139_policy.get("dateOfApplication"))
            _display("Requested Type", acord139_policy.get("requestedType"))

        with st.expander("2. Locations (acord139Location)", expanded=True):
            if not acord139_locations:
                st.markdown("- No location data found.")
            else:
                for index, location in enumerate(acord139_locations, start=1):
                    if not isinstance(location, dict):
                        st.markdown(f"- Location {index}: {location}")
                        continue
                    st.markdown(f"- **Location {index}**")
                    _display("Address", location.get("Address_of_Property"))
                    _display("Location #", location.get("location_number"))
                    _display("Building #", location.get("building_number"))
                    _display("Description", location.get("Description_of_Property"))
                    _display("Valuation", location.get("valuation"))
                    _display("Subject", location.get("subject"))
                    _display("Class Code", location.get("class_code"))
                    _display("100% Values", location.get("hundred_percent_value"))
                    _display("Limits of Insurance", location.get("limits_of_insurance"))
                    _display("Rate/Loss Cost", location.get("rate_or_loss_cause"))
                    _display("Premium", location.get("premium"))
                    _display("Total 100%", location.get("total_100_percent"))
                    _display("Total Rate of Loss Cost", location.get("total_rate_of_loss_cost"))
                    _display("Total Premium", location.get("total_premium"))

    else:
        # ACORD 125 display
        st.subheader("ACORD 125 Summary")
        st.markdown("Review the extracted ACORD 125 sections below.")

        with st.expander("1. Policy & General Information (acord125Policy)", expanded=True):
            _display("Agency Name", policy_header.get("agency") or policy_header.get("agency_name") or policy_header.get("producer"))
            _display("Date of Application", policy_header.get("form_date") or policy_header.get("date_of_application"))
            _display("Agency Phone Number", policy_header.get("agency_phone_number") or policy_header.get("producer_phone_number"))
            _display("Agency Address", policy_header.get("agency_address") or policy_header.get("producer_address"))
            _display("Status of Transaction", _summarize_status(policy_header.get("status_of_transaction")))
            _display_selected_lobs(lines_of_business)
            _display_selected_attachments(attachments)
            _display_from_sources("Proposed Effective Date", ["proposed_effective_date", "effective_date", "form_date"], [policy_information, policy_header])
            _display_from_sources("Proposed Expiration Date", ["proposed_expiration_date", "expiration_date"], [policy_information, policy_header])
            _display_from_sources("Billing Plan", ["billing_plan", "billing_plan_name"], [policy_information, policy_header])
            _display_from_sources("Payment Plan", ["payment_plan", "payment_plan_type"], [policy_information, policy_header])
            _display_from_sources("Method of Payment", ["method_of_payment", "payment_method", "payment_option"], [policy_information, policy_header])
            _display_from_sources("Audit Schedule", ["audit", "audit_schedule"], [policy_information, policy_header])
            _display_from_sources("Deposit Amount", ["deposit_amount", "deposit"], [policy_information, policy_header])
            _display_from_sources("Minimum Premium", ["minimum_premium", "minimum_premium_amount", "minimum_premium_total"], [policy_information, policy_header])
            _display_from_sources("Policy Premium", ["policy_premium", "premium", "policy_total_premium"], [policy_information, policy_header])
            _display_from_sources("SIC Code", ["sic"], [applicant, insured, policy_header])
            _display_from_sources("NAICS / Agency NAIC Code", ["naics"], [applicant, insured, policy_header])
            _display_from_sources("FEIN or Social Security Number", ["fein", "fein_or_soc_sec", "social_security_number"], [applicant, insured, policy_header])
            _display_from_sources("Website Address", ["website", "website_address"], [applicant, insured, policy_header])
            _display_selected_company_type(insured.get("company_type_options") or applicant.get("company_type_options") or insured.get("company_type") or applicant.get("company_type"))
            _display_from_sources("Insured Name", ["name", "insured_name", "applicant_name"], [insured, applicant, policy_header])
            _display_from_sources("Mailing Address", ["mailing_address", "address"], [insured, applicant, policy_header])
            _display_from_sources("License Number", ["license_number", "license_no", "license"], [insured, applicant, policy_header])

        with st.expander("2. Locations Schedule (acord125Location)", expanded=True):
            if not location_records:
                st.markdown("- No location data found.")
            else:
                for index, location in enumerate(location_records, start=1):
                    if not isinstance(location, dict):
                        st.markdown(f"- Location {index}: {location}")
                        continue
                    st.markdown(f"- **Location {index}**")
                    _display("Location Number", location.get("location_number") or location.get("location_no") or location.get("location") or location.get("loc_number"))
                    _display("Building Number", location.get("building_number") or location.get("bld_number") or location.get("building_no"))
                    _display("Full Address", _format_address(location.get("mailing_address") or location))
                    _display("Description of Operations", location.get("description_of_operations") or location.get("operations_description") or location.get("description"))
                    interest_value = (location.get("legal_interest") or location.get("owner_type") or location.get("occupancy_type"))
                    if not interest_value:
                        interests = []
                        if location.get("interest_owner") or location.get("owner"):
                            interests.append("Owner")
                        if location.get("interest_tenant") or location.get("tenant"):
                            interests.append("Tenant")
                        if location.get("interest_manager") or location.get("manager"):
                            interests.append("Manager")
                        if interests:
                            interest_value = ", ".join(interests)
                    _display("Interest", interest_value)
                    _display("Annual Revenues", location.get("annual_revenues") or location.get("annual_revenue"))
                    _display("Occupied Area", location.get("occupied_area") or location.get("occupied_sq_ft") or location.get("occupied_area_sq_ft"))
                    _display("Open to Public Area", location.get("open_to_public_area"))
                    _display("Total Building Area", location.get("total_building_area") or location.get("total_area"))
                    _display("Area Leased to Others", location.get("area_leased_to_others") or location.get("leased_area"))
                    _display("Nature of Business", location.get("nature_of_business") or location.get("business_nature"))
                    _display("Description of Primary Operations", location.get("description_of_primary_operations") or location.get("primary_operations_description") or location.get("primary_operations") or location.get("operations_description"))
                    _display("Number of Full-Time Employees", location.get("full_time_employees") or location.get("full_time_employee_count"))
                    _display("Number of Part-Time Employees", location.get("part_time_employees") or location.get("part_time_employee_count"))

        with st.expander("3. Prior Carrier History (acord125PriorCarrier)", expanded=True):
            if isinstance(prior_carrier, dict) and prior_carrier:
                _render_record_list("Prior carrier entries", [prior_carrier], [
                    ("Year", "year"),
                    ("Policy Category", "policy_category"),
                    ("Insurance Carrier Name", "carrier"),
                    ("Policy Number", "policy_number"),
                    ("Premium Amount", "annual_premium"),
                    ("Effective Date", "effective_date"),
                    ("Expiration Date", "expiration_date"),
                ])
            elif isinstance(prior_carrier, list) and prior_carrier:
                _render_record_list("Prior carrier entries", prior_carrier, [
                    ("Year", "year"),
                    ("Policy Category", "policy_category"),
                    ("Insurance Carrier Name", "carrier"),
                    ("Policy Number", "policy_number"),
                    ("Premium Amount", "annual_premium"),
                    ("Effective Date", "effective_date"),
                    ("Expiration Date", "expiration_date"),
                ])
            else:
                st.markdown("- No prior carrier history found.")

        with st.expander("4. Loss History (acord125LossHistory)", expanded=True):
            if not loss_history:
                st.markdown("- No loss history found.")
            else:
                for index, loss in enumerate(loss_history, start=1):
                    if not isinstance(loss, dict):
                        st.markdown(f"- Loss {index}: {loss}")
                        continue
                    st.markdown(f"- **Loss {index}**")
                    _display("Date of Occurrence", loss.get("date_of_occurrence") or loss.get("occurrence_date") or loss.get("date"))
                    _display("Type / Description of Occurrence or Claim", loss.get("description") or loss.get("type") or loss.get("claim_description"))
                    _display("Date of Claim", loss.get("date_of_claim") or loss.get("claim_date"))
                    _display("Amount Paid", loss.get("amount_paid") or loss.get("paid_amount") or loss.get("amount"))
                    _display("Amount Reserved", loss.get("amount_reserved") or loss.get("reserved_amount"))
                    _display("Subrogation Flag", loss.get("subrogation_flag") or loss.get("subrogation"))
                    _display("Claim Open Flag", loss.get("claim_open_flag") or loss.get("claim_open"))

    with st.expander("Raw JSON schema", expanded=False):
        st.json(result)

with col2:
    st.subheader("Uploaded document")
    st.write(f"Source: {source_name}")
    st.download_button(
        "Download uploaded file",
        uploaded_file.getvalue(),
        file_name=uploaded_file.name,
        mime=uploaded_file.type or "application/octet-stream",
    )

    if uploaded_file.type == "application/pdf" or Path(uploaded_file.name).suffix.lower() == ".pdf":
        pdf_bytes = uploaded_file.getvalue()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            temp_pdf_path = Path(tmp.name)

        try:
            with fitz.open(temp_pdf_path) as doc:
                images = []
                for page_number in range(doc.page_count):
                    page = doc.load_page(page_number)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    images.append(pix.tobytes("png"))
                st.image(images, caption=[f"Page {i+1}" for i in range(len(images))], use_column_width=True)
        except Exception:
            st.warning("PDF preview is not available in this browser. Please download the file to view it.")
        finally:
            if temp_pdf_path.exists():
                try:
                    temp_pdf_path.unlink(missing_ok=True)
                except PermissionError:
                    pass
    else:
        st.image(uploaded_file.getvalue(), caption=source_name)
