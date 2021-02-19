# NLP-Sarcasm-Detection
Neural Network Model for detecting sarcastic news headlines from text-corpus through BERT and LSTM

• Devised a neural network to accomplish context-based sentiment analysis for detecting sarcastic headlines from a text-based news headlines corpus.

• Used the Python NLTK library and Keras library to perform pre-processing on the text-corpus through techniques such as Tokenization, Stop-word removal, Lemmatization etc.

• Built a convolutional neural network system through the LSTM and the BERT models to perform classification tasks on the text-based input. The LSTM model was used with view to incorporate context within text-based input and the BERT model for optimal performance on the Natural Language Processing tasks.

• Performed hyperparameter tuning through optimizing factors such as Embedding Lengths, Dropout, Epoch-Frequency etc.

Key commands to invoke functionality:

python3 run_glue.py --model_type bert --model_name_or_path bert-base-uncased --task_name sarcasm --do_train --do_eval --do_test --do_lower_case --data_dir data --max_seq_length 32 --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 32 --per_gpu_test_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir output

#for testing only python3 run_glue.py --model_type bert --model_name_or_path bert-base-uncased --task_name sarcasm --do_test --do_lower_case --data_dir data --max_seq_length 32 --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 32 --per_gpu_test_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir output

#for training and eval python3 run_glue.py --model_type bert --model_name_or_path bert-base-uncased --task_name sarcasm --do_train --do_eval --do_lower_case --data_dir data --max_seq_length 32 --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 32 --per_gpu_test_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir output
