import pandas as pd


def read_local_data(data_dir):
    static_vars = pd.read_csv(data_dir + 'static_vars.csv')
    dynamic_vars = pd.read_csv(data_dir + 'dynamic_vars.csv')
    outcome_vars = pd.read_csv(data_dir + 'outcome_vars.csv')
    input_vars = pd.read_csv(data_dir + 'input_vars.csv')

    return static_vars, dynamic_vars, outcome_vars, input_vars

def read_all_data(data_dir):
    inpatients_records = pd.read_csv(
        data_dir + 'CDSL_01.csv', sep=',')
    inpatients_records.rename({'PATIENT ID': 'patient_id',
                               'EDAD/AGE': 'age',
                               'SEXO/SEX': 'sex',
                               'DIAG ING/INPAT': 'inpatient_covid_dx',
                               'F_INGRESO/ADMISSION_D_ING/INPAT': 'inpatient_admit_date',
                               'F_ENTRADA_UC/ICU_DATE_IN': 'icu_admit_date',
                               'F_SALIDA_UCI/ICU_DATE_OUT': 'icu_discharge_date',
                               'UCI_DIAS/ICU_DAYS': 'icu_days',
                               'F_ALTA/DISCHARGE_DATE_ING': 'inpatient_discharge_date',
                               'MOTIVO_ALTA/DESTINY_DISCHARGE_ING': 'discharge_destination',
                               'F_INGRESO/ADMISSION_DATE_URG/EMERG': 'emergency_visit_date',
                               'HORA/TIME_ADMISION/ADMISSION_URG/EMERG': 'emergency_visit_time',
                               'ESPECIALIDAD/DEPARTMENT_URG/EMERG': 'emergency_visit_department',
                               'DIAG_URG/EMERG': 'emergency_visit_dx',
                               'DESTINO/DESTINY_URG/EMERG': 'emergency_visit_destination',
                               'HORA/TIME_CONSTANT_PRIMERA/FIRST_URG/EMERG': 'emergency_first_vitals_record_time',
                               'TEMP_PRIMERA/FIRST_URG/EMERG': 'emergency_first_temperature',
                               'FC/HR_PRIMERA/FIRST_URG/EMERG': 'emergency_first_heart_rate',
                               'GLU_PRIMERA/FIRST_URG/EMERG': 'emergency_first_glucose',
                               'SAT_02_PRIMERA/FIRST_URG/EMERG': 'emergency_first_spo2',
                               'TA_MAX_PRIMERA/FIRST/EMERG_URG': 'emergency_first_max_bp',
                               'TA_MIN_PRIMERA/FIRST_URG/EMERG': 'emergency_first_min_bp',
                               'HORA/TIME_CONSTANT_ULTIMA/LAST_URG/EMERG': 'emergency_last_vitals_record_time',
                               'FC/HR_ULTIMA/LAST_URG/EMERG': 'emergency_last_temperature',
                               'TEMP_ULTIMA/LAST_URG/EMERG': 'emergency_last_heart_rate',
                               'GLU_ULTIMA/LAST_URG/EMERG': 'emergency_last_glucose',
                               'SAT_02_ULTIMA/LAST_URG/EMERG': 'emergency_last_spo2',
                               'TA_MAX_ULTIMA/LAST_URGEMERG': 'emergency_last_max_bp',
                               'TA_MIN_ULTIMA/LAST_URG/EMERG': 'emergency_last_min_bp',
                               }, axis=1, inplace=True)

    for timecol in ['inpatient_admit_date',
                    'inpatient_discharge_date',
                    'icu_admit_date',
                    'icu_discharge_date', ]:
        inpatients_records.loc[:, timecol] = pd.to_datetime(
            inpatients_records[timecol], dayfirst=True)

    drugs = pd.read_csv(data_dir + 'CDSL_04.csv', sep=';',
                        encoding='unicode_escape')
    drugs.rename({'IDINGRESO': 'patient_id',
                    'DRUG_COMERCIAL_NAME': 'drug_name',
                    'DAILY_AVRG_DOSE': 'average_daily_dose',
                    'DRUG_START_DATE': 'start_date',
                    'DRUG_END_DATE': 'end_date',
                    'ATC5_NAME': 'ATC5_description',
                    'ID_ATC5': 'ATC5_id',
                    'ATC7_NAME': 'ATC7_description',
                    'ID_ATC7': 'ATC7_id'}, axis=1, inplace=True)

    for timecol in ['start_date', 'end_date']:
        drugs.loc[:, timecol] = pd.to_datetime(
            drugs[timecol], dayfirst=True)

    # this next dataframe is not great, only (in my trials) gets data after icu discharge, like in
    # an observation period
    clinical_vars = pd.read_csv(
        data_dir + 'CDSL_02.csv', sep=';', encoding='unicode_escape')

    clinical_vars.rename({'IDINGRESO': 'patient_id',
                          'CONSTANTS_ING_DATE': 'record_date',
                          'CONSTANTS_ING_TIME': 'record_time',
                          'FC_HR_ING': 'heart_rate',
                          'GLU_GLY_ING': 'glucose',
                          'SAT_02_ING': 'spo2',
                          'SAT_02_ING_OBS': 'spo2_obs',
                          'TA_MAX_ING': 'max_bp',
                          'TA_MIN_ING': 'min_bp',
                          'TEMP_ING': 'temperature'}, axis=1, inplace=True)
    clinical_vars.loc[:, 'datetime'] = pd.to_datetime(clinical_vars['record_date'] + ' ' + clinical_vars['record_time'],
                                                      dayfirst=True)
    clinical_vars.drop(['record_date', 'record_time'], axis=1, inplace=True)

    # need to find better way to read this one
    # note this skips lots of lines with extra descriptions
        
    labs = pd.read_csv(data_dir + 'CDSL_06.csv', sep=';',
                        encoding='unicode_escape',
                        usecols=['IDINGRESO',
                                    'LAB_NUMBER',
                                    'LAB_DATE',
                                    'TIME_LAB',
                                    'ITEM_LAB',
                                    'VAL_RESULT',
                                    'UD_RESULT',
                                    'REF_VALUES', ])

    labs.rename({'IDINGRESO': 'patient_id',
                    'LAB_NUMBER': 'lab_request_id',
                    'LAB_DATE': 'lab_date',
                    'TIME_LAB': 'lab_time',
                    'ITEM_LAB': 'label',
                    'VAL_RESULT': 'value',
                    'UD_RESULT': 'uom',
                    'REF_VALUES': 'ref_values', }, axis=1, inplace=True)

    labs.loc[:, 'datetime'] = pd.to_datetime(labs['lab_date'] + ' ' + labs['lab_time'],
                                             dayfirst=True)
    labs.drop(['lab_date', 'lab_time'], axis=1, inplace=True)

    labs_v2 = pd.read_csv(data_dir + 'CDSL_06_v2.csv', sep=';',
                        encoding='unicode_escape',
                        usecols=['IDINGRESO',
                                    'LAB_NUMBER',
                                    'LAB_DATE',
                                    'TIME_LAB',
                                    'ITEM_LAB',
                                    'VAL_RESULT',
                                    'UD_RESULT',
                                    'REF_VALUES', ])

    labs_v2.rename({'IDINGRESO': 'patient_id',
                    'LAB_NUMBER': 'lab_request_id',
                    'LAB_DATE': 'lab_date',
                    'TIME_LAB': 'lab_time',
                    'ITEM_LAB': 'label',
                    'VAL_RESULT': 'value',
                    'UD_RESULT': 'uom',
                    'REF_VALUES': 'ref_values', }, axis=1, inplace=True)

    labs_v2.loc[:, 'datetime'] = pd.to_datetime(labs_v2['lab_date'] + ' ' + labs_v2['lab_time'],
                                             dayfirst=True)
    labs_v2.drop(['lab_date', 'lab_time'], axis=1, inplace=True)

    codes_emergency = pd.read_csv(
        data_dir + 'CDSL_03.csv', sep=';', encoding='unicode_escape')
    codes_emergency.rename({'IDINGRESO': 'patient_id'}, axis=1, inplace=True)

    codes_inpatient = pd.read_csv(
        data_dir + 'CDSL_05.csv', sep=';', encoding='unicode_escape')
    codes_inpatient.rename({'IDINGRESO': 'patient_id'}, axis=1, inplace=True)

    return inpatients_records, drugs, clinical_vars, labs, labs_v2, codes_emergency, codes_inpatient
