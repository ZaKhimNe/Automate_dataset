import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_error_distribution(csv_path):
    df = pd.read_csv(csv_path)
    
    models =['GPT', 'Gemini', 'Claude']
    error_reasons =['Visual_Miss', 'Pretrained_Overreliance', 'Rule_Ignored', 'Reasoning_Error', 'Hallucination']
    
    error_data =[]
    
    for model in models:
        df_wrong = df[df[f'{model}_Is_Correct'] == False]
        
        error_counts = df_wrong[f'{model}_Failure_Reason'].value_counts()
        
        total_errors = len(df_wrong)
        for reason in error_reasons:
            count = error_counts.get(reason, 0)
            percentage = (count / total_errors) * 100 if total_errors > 0 else 0
            error_data.append({'Model': model, 'Failure_Reason': reason, 'Percentage': percentage})

    df_plot = pd.DataFrame(error_data)
    
    pivot_df = df_plot.pivot(index='Model', columns='Failure_Reason', values='Percentage')
    
    # Dùng bảng màu học thuật đẹp mắt của Seaborn
    colors = sns.color_palette("Set2", len(error_reasons))
    
    ax = pivot_df.plot(kind='bar', stacked=True, figsize=(10, 6), color=colors, edgecolor='black')
    
    plt.title('Error Distribution When VLMs Fail (Phân phối bản chất lỗi)', fontsize=14, fontweight='bold')
    plt.xlabel('Vision-Language Models', fontsize=12)
    plt.ylabel('Percentage of Errors (%)', fontsize=12)
    plt.legend(title='Failure Reason', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('Error_Distribution.png', dpi=300)
    print("✅ Đã xuất biểu đồ Error_Distribution.png")


def plot_attack_vulnerability(csv_path):
    df = pd.read_csv(csv_path)
    models = ['GPT', 'Gemini', 'Claude']
    
    # Lấy top 5 loại Attack Type phổ biến nhất
    top_attacks = df['Attack_Type_1'].value_counts().head(5).index
    
    heatmap_data = pd.DataFrame(index=top_attacks, columns=models)
    
    for attack in top_attacks:
        df_attack = df[df['Attack_Type_1'] == attack]
        for model in models:
            # Tính Accuracy (%) của model đó trên loại bẫy này
            accuracy = df_attack[f'{model}_Is_Correct'].mean() * 100
            heatmap_data.loc[attack, model] = accuracy

    # Chuyển kiểu dữ liệu sang float để vẽ heatmap
    heatmap_data = heatmap_data.astype(float)
    
    plt.figure(figsize=(8, 6))
    # Dùng màu YlOrRd (Đỏ là nguy hiểm/thấp, Vàng nhạt là an toàn/cao)
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd_r", cbar_kws={'label': 'Accuracy (%)'})
    
    plt.title('Model Accuracy Across Different Attack Types', fontsize=14, fontweight='bold')
    plt.ylabel('Adversarial Tactics (Chiến thuật)', fontsize=12)
    plt.tight_layout()
    plt.savefig('Attack_Vulnerability_Heatmap.png', dpi=300)
    print("✅ Đã xuất biểu đồ Attack_Vulnerability_Heatmap.png")

if __name__ == "__main__":
    # Thay đường dẫn tới file Dataset.csv của bạn
    plot_error_distribution('../Dataset.csv')
    plot_attack_vulnerability()
