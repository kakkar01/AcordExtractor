"""
NuExtract3 JSON templates for common ACORD forms.

Each template uses the NuExtract field-type vocabulary:
  - "verbatim-string"   : copy-paste exactly from the document
  - "string"            : allow light paraphrasing / normalization
  - "integer"           : whole number
  - "number"            : integer or decimal
  - "date"              : ISO 8601 date  (YYYY-MM-DD)
  - "date-time"         : ISO 8601 date-time
  - "boolean"           : true / false
  - ["opt1","opt2",…]   : single-select enum
  - [["A","B","C"]]     : multi-select enum
"""

# ─── ACORD 25 — Certificate of Liability Insurance ───────────────────────────

ACORD_25 = {
    "form_info": {
        "form_number": "verbatim-string",
        "form_date": "date",
        "revision_date": "date",
    },
    "producer": {
        "name": "verbatim-string",
        "address": {
            "street": "verbatim-string",
            "city": "verbatim-string",
            "state": "verbatim-string",
            "zip": "verbatim-string",
        },
        "phone": "verbatim-string",
        "fax": "verbatim-string",
        "email": "verbatim-string",
        "contact_name": "verbatim-string",
    },
    "insured": {
        "name": "verbatim-string",
        "address": {
            "street": "verbatim-string",
            "city": "verbatim-string",
            "state": "verbatim-string",
            "zip": "verbatim-string",
        },
    },
    "coverages": [
        {
            "type": [
                "Commercial General Liability",
                "Automobile Liability",
                "Umbrella / Excess Liability",
                "Workers Compensation",
                "Other",
            ],
            "insurer": "verbatim-string",
            "naic_number": "verbatim-string",
            "policy_number": "verbatim-string",
            "effective_date": "date",
            "expiration_date": "date",
            "limits": {
                "each_occurrence": "number",
                "aggregate": "number",
                "products_completed_ops_aggregate": "number",
                "personal_advertising_injury": "number",
                "damage_to_rented_premises": "number",
                "medical_expense": "number",
                "combined_single_limit": "number",
                "bodily_injury_per_person": "number",
                "bodily_injury_per_accident": "number",
                "property_damage": "number",
                "each_accident_wc": "number",
                "disease_policy_limit": "number",
                "disease_each_employee": "number",
            },
        }
    ],
    "certificate_holder": {
        "name": "verbatim-string",
        "address": {
            "street": "verbatim-string",
            "city": "verbatim-string",
            "state": "verbatim-string",
            "zip": "verbatim-string",
        },
    },
    "description_of_operations": "verbatim-string",
    "cancellation_days_notice": "integer",
    "authorized_representative": "verbatim-string",
}

# ─── ACORD 125 — Commercial Insurance Application ────────────────────────────

ACORD_125 = {
    "form_info": {
        "form_number": "verbatim-string",
        "form_date": "date",
        "agency": "verbatim-string",
        "agency_customer_id": "verbatim-string",
    },
    "applicant": {
        "name": "verbatim-string",
        "mailing_address": {
            "street": "verbatim-string",
            "city": "verbatim-string",
            "state": "verbatim-string",
            "zip": "verbatim-string",
            "country": "country",
        },
        "gl_code": "verbatim-string",
        "sic": "verbatim-string",
        "naics": "verbatim-string",
        "fein": "verbatim-string",
        "business_type": [
            "Individual",
            "Partnership",
            "Corporation",
            "LLC",
            "Other",
        ],
        "years_in_business": "integer",
        "years_with_current_management": "integer",
        "website": "verbatim-string",
        "contact_name": "verbatim-string",
        "phone": "verbatim-string",
        "email": "verbatim-string",
    },
    "policy_information": {
        "proposed_effective_date": "date",
        "proposed_expiration_date": "date",
        "billing_plan": ["Agency Bill", "Direct Bill"],
        "payment_plan": "verbatim-string",
        "audit": ["Annual", "Semi-Annual", "Quarterly", "Monthly", "None"],
    },
    "coverages_requested": {
        "commercial_general_liability": "boolean",
        "commercial_auto": "boolean",
        "commercial_property": "boolean",
        "workers_compensation": "boolean",
        "umbrella_excess": "boolean",
        "inland_marine": "boolean",
        "crime": "boolean",
        "other": "verbatim-string",
    },
    "description_of_operations": "verbatim-string",
    "premises": [
        {
            "location_number": "integer",
            "street": "verbatim-string",
            "city": "verbatim-string",
            "state": "verbatim-string",
            "zip": "verbatim-string",
            "full_time_employees": "integer",
            "part_time_employees": "integer",
            "annual_revenues": "number",
        }
    ],
    "prior_losses": [
        {
            "year": "integer",
            "number_of_claims": "integer",
            "amount_paid": "number",
            "amount_reserved": "number",
        }
    ],
}

# ─── ACORD 130 — Workers Compensation Application ────────────────────────────

ACORD_130 = {
    "form_info": {
        "form_number": "verbatim-string",
        "form_date": "date",
    },
    "applicant": {
        "name": "verbatim-string",
        "fein": "verbatim-string",
        "mailing_address": {
            "street": "verbatim-string",
            "city": "verbatim-string",
            "state": "verbatim-string",
            "zip": "verbatim-string",
        },
        "years_in_business": "integer",
        "business_type": [
            "Individual",
            "Partnership",
            "Corporation",
            "LLC",
            "Other",
        ],
    },
    "policy_information": {
        "proposed_effective_date": "date",
        "proposed_expiration_date": "date",
        "state": "verbatim-string",
        "experience_mod": "number",
        "experience_mod_effective_date": "date",
    },
    "class_codes": [
        {
            "class_code": "verbatim-string",
            "description": "verbatim-string",
            "state": "verbatim-string",
            "number_of_employees": "integer",
            "annual_remuneration": "number",
            "estimated_annual_premium": "number",
        }
    ],
    "prior_carrier_information": {
        "carrier": "verbatim-string",
        "policy_number": "verbatim-string",
        "effective_date": "date",
        "expiration_date": "date",
        "annual_premium": "number",
    },
    "loss_history": [
        {
            "policy_year": "integer",
            "number_of_claims": "integer",
            "total_incurred": "number",
        }
    ],
}

# ─── ACORD 140 — Property Section (custom schema provided by user) ──────────

ACORD_140 = {
    "acord140Policy": {
        "agency": "Agency Name",
        "carrier": "Carrier",
        "naicCode": "naic",
        "effectiveDate": "Effective Date",
        "applicantFirstNamedInsured": "Insured Name",
        "constructionTypePremisesInformation": "ConstructionType.1",
        "constructionTypeAdditionalPremises": ["ConstructionType.2", "ConstructionType.3"],
        "fireDistrictPremisesInformation": "firedistrict.1",
        "fireDistrictAdditionalPremises": "Firedistrict.2",
        "distanceToFireStationPremisesInformation": "distanceFirestation.1",
        "distanceToFireStationAdditionalPremises": "distanceFirestation.2",
        "distanceToHydrantPremisesInformation": "distanceHydrant.1",
        "distanceToHydrantAdditionalPremises": "distanceHydrant.2",
        "protClassPremisesInformation": "protcl.1",
        "protClassAdditionalPremises": ["protcl.2", "protcl.3"],
        "numberStoriesPremisesInformation": "stories#.1",
        "numberStoriesAdditionalPremises": ["stories#.2", "stories#.3"],
        "numberBasementsPremisesInformation": "Basements#.1",
        "numberBasementsAdditionalPremises": ["Basements#.2", "Basements#.3"],
        "yearBuiltPremisesInformation": "yearBuilt.1",
        "yearBuiltAdditionalPremises": ["yearBuilt.2", "yearBuilt.3"],
        "totalAreaPremisesInformation": "totalArea.1",
        "totalAreaAdditionalPremises": ["totalArea.2", "totalArea.3"],
        "roofingyearPremisesInformation": "YEAR from BuildingImprovements.1[] where TYPE ~ 'ROOFING' and SELECTION INDICATOR == ':selected:'",
        "heatingYearPremisesInformation": "YEAR from BuildingImprovements.1[] where TYPE ~ 'HEATING' and SELECTION INDICATOR == ':selected:'",
        "wiringYearPremisesInformation": "YEAR from BuildingImprovements.1[] where TYPE ~ 'WIRING' and SELECTION INDICATOR == ':selected:'",
        "plumbingYearPremisesInformation": "YEAR from BuildingImprovements.1[] where TYPE ~ 'PLUMBING' and SELECTION INDICATOR == ':selected:'",
        "roofingYearAdditionalPremises": "YEAR from (BuildingImprovements.2[] + BuildingImprovements.3[]) where TYPE ~ 'ROOFING' and SELECTION INDICATOR == ':selected:'",
        "heatingYearAdditionalPremises": "YEAR from (BuildingImprovements.2[] + BuildingImprovements.3[]) where TYPE ~ 'HEATING' and SELECTION INDICATOR == ':selected:'",
        "wiringYearAdditionalPremises": "YEAR from (BuildingImprovements.2[] + BuildingImprovements.3[]) where TYPE ~ 'WIRING' and SELECTION INDICATOR == ':selected:'",
        "plumbingYearAdditionalPremises": "YEAR from (BuildingImprovements.2[] + BuildingImprovements.3[]) where TYPE ~ 'PLUMBING' and SELECTION INDICATOR == ':selected:'",
        "roofTypePremisesInformation": "roofType.1",
        "roofTypeAdditionalPremises": "roofType.2",
        "otherOccupanciesPremisesInformation": "otherOccupancies.1",
        "otherOccupanciesAdditionalPremises": ["otherOccupancies.2", "otherOccupancies.3"],
        "heatingSourceInclWoodburningStoveOrFirePlaceInsertPremisesInformation": "HeatingSourceInclWoodburningStoveFireplace.1",
        "heatingSourceInclWoodburningStoveOrFirePlaceInsertAdditionalPremises": "HeatingSourceInclWoodburningStoveFireplace.2",
        "primaryHeatPremisesInformation": "PrimaryHeat.1[].TYPE (where SELECTION INDICATOR == ':selected:')",
        "primaryHeatAdditionalPremises": "PrimaryHeat.2[].TYPE (where SELECTION INDICATOR == ':selected:')",
        "secondaryHeatPremisesInformation": "SecondaryHeat.1[].TYPE (where SELECTION INDICATOR == ':selected:')",
        "secondaryHeatAdditionalPremises": "SecondaryHeat.2[].TYPE (where SELECTION INDICATOR == ':selected:')",
        "burglarAlarmTypePremisesInformation": "BurglarAlarmType.1",
        "burglarAlarmTypeAdditionalPremises": ["BurglarAlarmType.2", "BurglarAlarmType.3"],
        "premisesFireProtectionPremisesInformation": "premisesFireProtection.1",
        "premisesFireProtectionAdditionalPremises": ["premisesFireProtection.2", "premisesFireProtection.3"],
        "percentSprinklerPremisesInformation": "Sprnk%.1",
        "percentSprinklerAdditionalPremises": "Sprnk%.2",
    },

    "acord140PremisesInformation": {
        "subject_of_insurance": [
            "TABLES[where headers contain 'SUBJECT OF INSURANCE'].rows[].column[SUBJECT OF INSURANCE]",
            "Extract from table with headers: SUBJECT OF INSURANCE, AMOUNT, COINS %, VALUATION, CAUSES OF LOSS, etc.",
        ],
        "amount": [
            "TABLES[where headers contain 'SUBJECT OF INSURANCE'].rows[].column[AMOUNT]"
        ],
        "coins_percent": [
            "TABLES[where headers contain 'SUBJECT OF INSURANCE'].rows[].column[COINS %]"
        ],
        "valuation": [
            "TABLES[where headers contain 'SUBJECT OF INSURANCE'].rows[].column[VALUATION]"
        ],
        "causes_of_losses": [
            "TABLES[where headers contain 'SUBJECT OF INSURANCE'].rows[].column[CAUSES OF LOSS]"
        ],
        "inflation_guard_percent": [
            "TABLES[where headers contain 'SUBJECT OF INSURANCE'].rows[].column[INFLATION GUARD %]"
        ],
        "ded": [
            "TABLES[where headers contain 'SUBJECT OF INSURANCE'].rows[].column[DEDUCTIBLE or DED]"
        ],
        "ded_type": [
            "TABLES[where headers contain 'SUBJECT OF INSURANCE'].rows[].column[DED TYPE]"
        ],
        "blkt_number": [
            "TABLES[where headers contain 'SUBJECT OF INSURANCE'].rows[].column[BLKT #]"
        ],
        "forms_and_conditions_to_apply": [
            "TABLES[where headers contain 'SUBJECT OF INSURANCE'].rows[].column[FORMS AND CONDITIONS TO APPLY or FORMS AND CONDITIONS APPLY]"
        ],
        "premises_number": [
            "TABLES[where headers contain 'SUBJECT OF INSURANCE'].rows[].column[0] (first column - premises identifier)",
            "Also check FIELDS: premises_number.1, premises_number.2, premises_number.3",
        ],
        "building_number": [
            "FIELDS: building_number.1, building_number.2, building_number.3"
        ],
        "street_address": [
            "FIELDS: street_address.1, street_address.2"
        ],
        "bldg_desc": [
            "FIELDS: building_description.1, building_description.2"
        ],
    },

    "acord140BlanketSummary": {
        "blkt_number": "Blanket Information[].BLKT #",
        "amount": "Blanket Information[].AMOUNT",
        "blkt_type": "Blanket Information[].TYPE",
    },
}

# ─── ACORD 126 — General Liability Application ──────────────────────────────

ACORD_126 = {
    "acord126": {
        "agency": "verbatim-string",
        "carrier": "verbatim-string",
        "naicCode": "verbatim-string",
        "policyNumber": "verbatim-string",
        "effectiveDate": "date",
        "applicantFirstNamedInsured": "verbatim-string",
        "coveragesClaimsMadeOrOccurrence": [["Coverage_occurrence", "Coverage_ClaimsMade"]],
        "ProductsPremium": "number",
        "OtherPremium": "number",
        "PremisesOperationsPremium": "number",
        "TotalPremium": "number",
        "limitsappliesper": [["Limit_applies_per.policy", "Limit_applies_per.project", "Limit_applies_per.Location", "Limit_applies_per.Other"]],
        "GeneralAggregateLimitType": "number",
        "ProductsCompletedOperationsAggregateLimitType": "number",
        "PersonalAdvertisingInjuryLimitType": "number",
        "EachOccurenceLimitType": "number",
        "DamageToRentedPremisesLimitType": "number",
        "MedicalExpenseLimitType": "number",
        "EmployeeBenefitsLimitType": "number",
        "OtherLimitType": "verbatim-string",
        "OtherLimitAmount": "number",
        "deductibletypeandamount": "verbatim-string",
        "DeductiblesPropertyDamage": "number",
        "DeductiblesBodilyInjury": "number",
        "DeductiblesOther": "number",
        "proposedRetroactiveDate": "date",
        "entryDateIntoUninterruptedClaimsMadeCoverage": "date",
        "exclusions": "verbatim-string",
        "wasTailCoveragePurchased": "boolean",
        "deductiblePerClaim": "number",
        "numberOfEmployeesCoveredByEmployeeBenefitsPlans": "integer",
        "numberOfEmployees": "integer",
        "retroactiveDate": "date",
        "license": "verbatim-string",
    },
    "acord126Hazards": [
        {
            "location_number": "verbatim-string",
            "hazard_number": "verbatim-string",
            "classification_class": "verbatim-string",
            "class_code": "verbatim-string",
            "premium_basis": "verbatim-string",
            "exposure": "verbatim-string",
            "rate_prem_ops": "number",
            "rate_products": "number",
        }
    ],
}

# ─── ACORD 131 — Umbrella / Excess Liability Application ──────────────────────

ACORD_131 = {
    "acord131Policy": {
        "license": "verbatim-string",
        "naic_code": "verbatim-string",
        "agency": "verbatim-string",
        "policy_number": "verbatim-string",
        "named_insured": "verbatim-string",
        "carrier": "verbatim-string",
        "effective_date": "date",
        "transaction_type": ["New Business", "Renewal", "Endorsement", "Cancellation"],
        "retroactivedate_proposed": "date",
        "retroactivedate_current": "date",
        "agg_limit_liability": "number",
        "prod_comp_ops_limit_liability": "number",
        "ea_occ_limit_liability": "number",
        "retained_limit": "number",
        "first_dollar_defense": "boolean",
        "expiring_policy_number": "verbatim-string",
        "limit_of_insurance": "number",
        "aggregate_limit_for_ebl": "number",
        "retained_limit_for_ebl": "number",
        "retroactive_date_for_ebl": "date",
        "name_of_benefit_program": "verbatim-string",
    },
    "acord131Location": [
        {
            "location_number": "verbatim-string",
            "location_name": "verbatim-string",
            "annual_payroll": "number",
            "annual_grossSales": "number",
            "foreign_grossSales": "number",
            "number_employees": "integer",
            "name": "verbatim-string",
            "description": "verbatim-string",
        }
    ],
    "acord131UnderlyingInsurance": [
        {
            "type": ["AutomobileLiability", "GeneralLiability", "EmployersLiability"],
            "CarrierPolicyNumber": "verbatim-string",
            "policy_eff_date": "date",
            "policy_exp_date": "date",
            "cslea_acc__limit_amount": "number",
            "biea_acc_limit_amount": "number",
            "biea_per_limit_amount": "number",
            "pdea_acc_limit_amount": "number",
            "automobile_liability_annual_renewal_premium": "number",
            "rating_mod": "string",
            "generalliability_policy_type": [["occur", "claimsMade"]],
            "each_occurrence_limit_amount": "number",
            "general_aggr_limit_amount": "number",
            "personal_and_adv_injury_limit_amount": "number",
            "damage_to_rented_premises_limit_amount": "number",
            "medical_expense_limit_amount": "number",
            "prod_comp_op_aggregate_limit_amount": "number",
            "generalliability_prem_ops": "number",
            "generalliability_products": "number",
            "generalliability_other": "number",
            "each_accident_limit_amount": "number",
            "disease_each_employee_limit_amount": "number",
            "disease_policy_limit_limit_amount": "number",
            "employersliability_annual_renewal_premium": "number",
        }
    ],
    "extraUnderlyingInsurance": [
        {
            "type": "verbatim-string",
            "CarrierPolicyNumber": "verbatim-string",
            "policy_eff_date": "date",
            "policy_exp_date": "date",
            "other_limit_type": "verbatim-string",
            "other_limit_amount": "number",
            "other_annual_renewal_premium": "number",
            "rating_mod": "string",
        }
    ],
}

# ─── ACORD 139 — Commercial Property Application ────────────────────────────

ACORD_139 = {
    "acord139": {
        "named_insured": "verbatim-string",
        "AgencyNameAddress": "verbatim-string",
        "carrier": "verbatim-string",
        "effective_date": "date",
        "naic_code": "verbatim-string",
        "contact_name": "verbatim-string",
        "phone": "verbatim-string",
        "email_address": "verbatim-string",
        "coins_percent": "number",
        "Applicable_Cause_Of_Loss": "verbatim-string",
        "dateOfApplication": "date",
        "requestedType": "verbatim-string",
    },
    "acord139Location": [
        {
            "Address_of_Property": "verbatim-string",
            "location_number": "verbatim-string",
            "building_number": "verbatim-string",
            "Description_of_Property": "verbatim-string",
            "valuation": "number",
            "subject": "verbatim-string",
            "class_code": "verbatim-string",
            "hundred_percent_value": "number",
            "limits_of_insurance": "number",
            "rate_or_loss_cause": "number",
            "premium": "number",
            "total_100_percent": "number",
            "total_rate_of_loss_cost": "number",
            "total_premium": "number",
        }
    ],
}

# ─── Registry ────────────────────────────────────────────────────────────────

#: Map a human-friendly form name to its NuExtract template dict.
ACORD_SCHEMAS: dict[str, dict] = {
    "ACORD 25": ACORD_25,
    "ACORD 125": ACORD_125,
    "ACORD 126": ACORD_126,
    "ACORD 130": ACORD_130,
    "ACORD 131": ACORD_131,
    "ACORD 139": ACORD_139,
    "ACORD 140": ACORD_140,
}
