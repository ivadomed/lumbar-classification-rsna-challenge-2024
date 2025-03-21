from  inference_nfn import *
from inference_scs import *
from inference_sas import *



def generate_empty_csv(list_subjects):
    # Define the headers
    headers = [
        "row_id",
        "normal_mild",
        "moderate",
        "severe"
    ]

    # Define the base row IDs without the subject ID
    base_row_ids = [
        "left_neural_foraminal_narrowing_l1_l2",
        "left_neural_foraminal_narrowing_l2_l3",
        "left_neural_foraminal_narrowing_l3_l4",
        "left_neural_foraminal_narrowing_l4_l5",
        "left_neural_foraminal_narrowing_l5_s1",
        "left_subarticular_stenosis_l1_l2",
        "left_subarticular_stenosis_l2_l3",
        "left_subarticular_stenosis_l3_l4",
        "left_subarticular_stenosis_l4_l5",
        "left_subarticular_stenosis_l5_s1",
        "right_neural_foraminal_narrowing_l1_l2",
        "right_neural_foraminal_narrowing_l2_l3",
        "right_neural_foraminal_narrowing_l3_l4",
        "right_neural_foraminal_narrowing_l4_l5",
        "right_neural_foraminal_narrowing_l5_s1",
        "right_subarticular_stenosis_l1_l2",
        "right_subarticular_stenosis_l2_l3",
        "right_subarticular_stenosis_l3_l4",
        "right_subarticular_stenosis_l4_l5",
        "right_subarticular_stenosis_l5_s1",
        "spinal_canal_stenosis_l1_l2",
        "spinal_canal_stenosis_l2_l3",
        "spinal_canal_stenosis_l3_l4",
        "spinal_canal_stenosis_l4_l5",
        "spinal_canal_stenosis_l5_s1"
    ]

    # Create the rows with the specified values
    rows = [headers]
    for subject in list_subjects:
        for base_id in base_row_ids:
            row_id = f"{subject}_{base_id}"
            rows.append([row_id, 0., 0., 0.])

    return rows 

def eval(list_subjects, data_dir, result, model_nfn, model_scs, model_sas): 

    pred_nfn = eval_nfn(list_subjects, data_dir, model_nfn)
    pred_scs = eval_scs(list_subjects, data_dir, model_scs)
    pred_sas = eval_sas(list_subjects, data_dir, model_sas)

    for label, output in pred_nfn: 
        result.loc[result["row_id"] == label, ['normal_mild', 'moderate', 'severe']] = output
    for label, output in pred_scs: 
        result.loc[result["row_id"] == label, ['normal_mild', 'moderate', 'severe']] = output
    for label, output in pred_sas: 
        result.loc[result["row_id"] == label, ['normal_mild', 'moderate', 'severe']] = output
   

def main():

    data_dir = "../nrc-lumbar-balgrist-copy"
    list_subjects = [subject for subject in os.listdir(data_dir) if 'sub' in subject]
    list_subjects.sort()
    if list_subjects == []:
        return 'make sure to input the good folder because no subject was found'
    
    result = generate_empty_csv(list_subjects)
    result = pd.DataFrame(result[1:], columns=result[0])
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_nfn = ResNet(
        block="bottleneck",
        layers=[3, 4, 6, 3],
        block_inplanes=[64, 128, 256, 512],
        spatial_dims=3,
        n_input_channels=2,
        num_classes=3,
    ).cuda()

    chkpth = "models/nfn.pth"
    model_nfn.load_state_dict(torch.load(chkpth, map_location=device))


    model_scs = ResNet(
            block="bottleneck",
            layers=[3, 4, 6, 3],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=1,
            num_classes=3,
            ).cuda()

    chkpth = "models/scs.pth"  
    model_scs.load_state_dict(torch.load(chkpth, map_location=device))


    model_sas = ResNet(
            block="bottleneck",
            layers=[3, 4, 6, 3],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=1,
            num_classes=3,
            ).cuda()

    chkpth = "models/sas.pth" 
    model_sas.load_state_dict(torch.load(chkpth, map_location=device))

    eval(list_subjects, data_dir,result, model_nfn, model_scs, model_sas)
    
    
    result.to_csv("submission.csv", index=False)
    print(result.head())

if __name__ == "__main__":
    main()  