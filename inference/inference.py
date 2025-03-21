import inference_nfn
import inference_scs
import inference_sas

def eval(list_subjects, data_dir): 

    pred_nfn = eval_nfn(list_subjects, data_dir)
    pred_scs = eval_scs(list_subjects, data_dir)
    pred_sas = eval_sas(list_subjects, data_dir)

    for label, output in pred_nfn: 
        result.loc[result["row_id"] == label, ['normal_mild', 'moderate', 'severe']] = output
    for label, output in pred_scs: 
        result.loc[result["row_id"] == label, ['normal_mild', 'moderate', 'severe']] = output
    for label, output in pred_sas: 
        result.loc[result["row_id"] == label, ['normal_mild', 'moderate', 'severe']] = output
   

    