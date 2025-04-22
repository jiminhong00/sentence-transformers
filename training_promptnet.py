import logging
import traceback

import torch
from datasets import load_dataset

from sentence_transformers.cross_encoder.CrossEncoder_multiple import CrossEncoder_multiple
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
from sentence_transformers.cross_encoder.losses import ListOrderLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

debug = False

def main():
    model_name = "allenai/longformer-base-4096"

    # Set the log level to INFO to get more information
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    # train_batch_size and eval_batch_size inform the size of the batches, while mini_batch_size is used by the loss
    # to subdivide the batch into smaller parts. This mini_batch_size largely informs the training speed and memory usage.
    # Keep in mind that the loss does not process `train_batch_size` pairs, but `train_batch_size * num_docs` pairs.
    train_batch_size = 1
    eval_batch_size = 1
    mini_batch_size = 1
    num_epochs = 1
    max_docs = None
    respect_input_order = True  # Whether to respect the original order of documents

    # 1. Define our CrossEncoder model
    # Set the seed so the new classifier weights are identical in subsequent runs
    torch.manual_seed(12)
    model = CrossEncoder_multiple(model_name, num_labels=1)
    print("Model max length:", model.max_length)
    print("Model num labels:", model.num_labels)

    # 2. Load the MS MARCO dataset: https://huggingface.co/datasets/microsoft/ms_marco
    logging.info("Read train dataset")
    # Change Here dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train")
    dataset = load_dataset("json", data_files="./dataset/rerank/after_reorder/formatted/uprise_task_gen_to_send.json")["train"]
    if(debug):
        print(type(dataset))
        print("len", len(dataset))
    # print(dataset.keys())
    def qp_mapper(batch):
        # print(batch)
        processed_queries = []
        processed_p1 = []
        processed_p2 = []
        processed_p3 = []
        processed_labels = []
        label_value = [1.0, 0.7, 0.3, 0.0]
        result = {"query": [], "p1": [], "p2": [], "p3": [], "labels" : []}
        for query, prompts, order  in zip(batch["query"], batch["perm_list"], batch["perm_order"]):
            processed_queries.append(query)
            processed_p1.append(prompts[0])
            processed_p2.append(prompts[1])
            processed_p3.append(prompts[2])
            temp_label = [0,0,0,0]
            for i in range(len(order)):
                temp_label[order[i]] = label_value[i]
            temp_label[0] = label_value[len(order)]
            processed_labels.append(temp_label)
        return {
            "query": processed_queries,
            "p1": processed_p1,
            "p2": processed_p2,
            "p3": processed_p3,
            "labels": processed_labels,
        }
    
    def listwise_mapper(batch, max_docs: int | None = 10):
        processed_queries = []
        processed_docs = []
        processed_labels = []

        for query, passages_info in zip(batch["query"], batch["passages"]):
            # Extract passages and labels
            passages = passages_info["passage_text"]
            labels = passages_info["is_selected"]

            # Pair passages with labels and sort descending by label (positives first)
            paired = sorted(zip(passages, labels), key=lambda x: x[1], reverse=True)

            # Separate back to passages and labels
            sorted_passages, sorted_labels = zip(*paired) if paired else ([], [])

            # Filter queries without any positive labels
            if max(sorted_labels) < 1.0:
                continue

            # Truncate to max_docs
            if max_docs is not None:
                sorted_passages = list(sorted_passages[:max_docs])
                sorted_labels = list(sorted_labels[:max_docs])

            processed_queries.append(query)
            processed_docs.append(sorted_passages)
            processed_labels.append(sorted_labels)

        return {
            "query": processed_queries,
            "docs": processed_docs,
            "labels": processed_labels,
        }

    # Create a dataset with a "query" column with strings, a "docs" column with lists of strings,
    # and a "labels" column with lists of floats
    if(debug): 
        print(dataset.column_names)
        print(type(dataset))
    dataset = dataset.map(
        lambda batch: qp_mapper(batch=batch),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Processing listwise samples",
    )
    # print(type(dataset))
    # print(dataset.column_names)

    dataset = dataset.train_test_split(test_size=1_00)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    logging.info(train_dataset)

    # 3. Define our training loss
    loss = ListOrderLoss(model, mini_batch_size=mini_batch_size, respect_input_order=respect_input_order)

    # 4. Define the evaluator. We use the CENanoBEIREvaluator, which is a light-weight evaluator for English reranking
    # evaluator = CrossEncoderNanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"], batch_size=eval_batch_size)
    # evaluator(model)

    # 5. Define the training arguments
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"promptnet-{short_model_name}-listorderloss"
    args = CrossEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        load_best_model_at_end=True,
        metric_for_best_model="eval_NanoBEIR_R100_mean_ndcg@10",
        # Optional tracking/debugging parameters:
        eval_strategy="no",
        eval_steps=500,
        save_strategy="no",
        save_steps=500,
        save_total_limit=2,
        logging_steps=250,
        logging_first_step=True,
        run_name=run_name+"1",  # Will be used in W&B if `wandb` is installed
        seed=12,
        gradient_accumulation_steps = 8 # b/c of small batch size
    )

    # 6. Create the trainer & start training
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        # evaluator=evaluator,
    )
    trainer.train()

    # 7. Evaluate the final model, useful to include these in the model card
    # evaluator(model)

    # 8. Save the final model
    final_output_dir = f"models/{run_name}/final"
    model.save_pretrained(final_output_dir)

   

if __name__ == "__main__":
    main()
