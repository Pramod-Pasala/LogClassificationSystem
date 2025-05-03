from processor_regex import classify_with_regex
from processor_bert import classify_with_bert
from processor_llm import classify_with_llm
import pandas as pd

def classify_csv(input_file,output_file=None,server=False):
    df = pd.read_csv(input_file)
    df['label'] = classify(list(zip(df['source'], df['log_message'])))
    if not server:
        if output_file is None:
            output_file = input_file.replace('.csv', '_classified.csv')
        df.to_csv(output_file, index=False)
    else:
        return df
def classify_log(source,log_message):
    if source == "LegacyCRM":
        return classify_with_llm(log_message)
    else:
        label = classify_with_regex(log_message)     
        return label if label else classify_with_bert(log_message)
   

def classify(logs):
    labels =[]
    for source,log_message in logs:
        label = classify_log(source,log_message)
        labels.append(label)
    return labels


if __name__ == "__main__":
    
    logs = [
        ("ModernCRM", "IP 192.168.133.114 blocked due to potential attack"),
        ("BillingSystem", "User User12345 logged in."),
        ("AnalyticsEngine", "File data_6957.csv uploaded successfully by user User265."),
        ("AnalyticsEngine", "Backup completed successfully."),
        ("ModernHR", "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1 RCODE  200 len: 1583 time: 0.1878400"),
        ("ModernHR", "Admin access escalation detected for user 9429"),
        ("LegacyCRM", "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."),
        ("LegacyCRM", "Invoice generation process aborted for order ID 8910 due to invalid tax calculation module."),
        ("LegacyCRM", "The 'BulkEmailSender' feature is no longer supported. Use 'EmailCampaignManager' for improved functionality."),
        ("LegacyCRM", " The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025")
    ]


    classified_logs = classify(logs)
    for log, label in zip(logs, classified_logs):
        print(f"Log: {log[1]} - Classified as: {label}")