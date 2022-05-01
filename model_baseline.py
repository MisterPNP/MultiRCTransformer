import transformers
import sklearn


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = sklearn.metrics.accuracy_score(labels, preds)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(labels, preds).ravel()
    return {'accuracy': acc, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}


def get_model(use_cuda=True):
    model = transformers.AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)
    if use_cuda:
        model = model.to('cuda')
    return model


def get_trainer(model, train_dataset, test_dataset, use_cuda=True):
    return transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=transformers.TrainingArguments(
            output_dir='./results',  # output directory
            num_train_epochs=3,  # total number of training epochs
            per_device_train_batch_size=64,  # batch size per device during training
            per_device_eval_batch_size=64,  # batch size for evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            # load_best_model_at_end=True,
            # -> When set to True, the parameter save_strategy needs to be the same as eval_strategy,
            #    and in the case it is “steps”, save_steps must be a round multiple of eval_steps.
            # metric_for_best_model='eval_tp',  # defualts to 'loss'
            logging_strategy='steps',
            logging_steps=200,  # log each logging_steps
            logging_dir='./logs',  # directory for storing logs
            save_strategy='no',
            # save_steps=200,  # save weights each save_steps
            evaluation_strategy='steps',
            eval_steps=200,  # evaluate each eval_steps
            # log_level='debug',
            # skip_memory_metrics=False,
            gradient_checkpointing=True,  # saves memory, might be slower
            no_cuda=not use_cuda,  # force trainer to use CPU
            # fp16=True,  # use half-precision floats
        ),
        compute_metrics=compute_metrics,
    )
