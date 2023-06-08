import numpy as np
import sklearn.metrics

# def calculate_auc(image_features_norm, text_features_norm, metadata):

#     similarity = (
#         image_features_norm.float() @ text_features_norm.float().T
#     )  # (batch_size, num_features) @ (num_features, num_prompts) = (batch_size, num_prompts)
#     if similarity.shape[1] > 1:
#         similarity_per_prompt = similarity.softmax(
#             dim=0
#         )  # (batch_size, num_prompts) -> (batch_size, num_prompts)
#         similarity_ensemble = similarity_per_prompt.mean(
#             dim=1
#         ).numpy()  # (batch_size, num_prompts) -> (batch_size, 1)
#     else:
#         # (batch_size, 1)
#         similarity_ensemble = similarity_per_prompt.numpy()
#     assert len(similarity_ensemble.shape) == 1

#     y_score = similarity_ensemble
#     y_true = metadata
#     if y_true.sum() > 0:
#         auc = sklearn.metrics.roc_auc_score(
#             y_true=y_true[~y_true.isnull()].values.astype(int),
#             y_score=y_score[~y_true.isnull()],
#             average="macro",
#             sample_weight=None,
#             max_fpr=None,
#             multi_class="raise",
#             labels=None,
#         )
#     else:
#         auc = np.nan
#     return auc


def skincon_calcualte_auc_all(image_features, text_features_dict, metadata_all):
    assert len(image_features) == len(
        metadata_all
    ), f"{len(image_features)} != {len(metadata_all)}"

    concept_cols = metadata_all.columns[metadata_all.columns.str.startswith("skincon_")]
    concept_cols = concept_cols.delete(
        concept_cols.tolist().index("skincon_Do not consider this image")
    )
    concept_cols = concept_cols.delete(concept_cols.tolist().index("skincon_Unnamed: 0"))

    image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
    auc_dict = {}
    for concept_col in concept_cols:
        text_features = text_features_dict[concept_col]
        text_features_norm = text_features / text_features.norm(dim=1, keepdim=True)

        similarity = (
            image_features_norm.float() @ text_features_norm.float().T
        )  # (batch_size, num_features) @ (num_features, num_prompts) = (batch_size, num_prompts)
        if similarity.shape[1] > 1:
            similarity_per_prompt = similarity.softmax(
                dim=0
            )  # (batch_size, num_prompts) -> (batch_size, num_prompts)
            similarity_ensemble = similarity_per_prompt.mean(
                dim=1
            ).numpy()  # (batch_size, num_prompts) -> (batch_size)
        else:
            # (batch_size, 1)
            similarity_ensemble = similarity[:, 0].numpy()
        assert len(similarity_ensemble.shape) == 1

        y_score = similarity_ensemble
        y_true = metadata_all[concept_col]
        if y_true.sum() > 0:
            auc = sklearn.metrics.roc_auc_score(
                y_true=y_true[~y_true.isnull()].values.astype(int),
                y_score=y_score[~y_true.isnull()],
                average="macro",
                sample_weight=None,
                max_fpr=None,
                multi_class="raise",
                labels=None,
            )
        else:
            auc = np.nan
        auc_dict[concept_col] = auc

        # auc = calculate_auc(
        #     image_features_norm=image_features_norm,
        #     text_features_norm=text_features_norm,
        #     metadata=metadata_all[concept_col],
        # )
        # auc_dict[concept_col] = auc
    return auc_dict
