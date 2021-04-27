import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import FreqDist
from wordcloud import WordCloud, STOPWORDS

sns.set(style='darkgrid', font_scale=2)


def plot_class_distribution(df):
    plt.figure(figsize=(12, 8))
    matplotlib.style.use('seaborn-poster')
    total_article_count = len(df)
    reliable_article_count = len(df[df['label'] == 0])
    unreliable_article_count = len(df[df['label'] == 1])
    distribution = [reliable_article_count / total_article_count,
                    unreliable_article_count / total_article_count]
    class_distribution = {
        'Article Types': ['Reliable', 'Unreliable'],
        'Percentage': distribution
    }
    class_distribution = pd.DataFrame(data=class_distribution)
    plot = sns.barplot('Article Types', 'Percentage',
                       data=class_distribution, palette=('muted'))
    for p in plot.patches:
        plot.annotate(format(p.get_height(), '.4f'),
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='center',
                      size=15,
                      xytext=(0, -15),
                      textcoords='offset points')
    plt.title('News Article Dataset Class Distribution')
    plt.savefig('./Figures/class_distribution.png')


def plot_word_cloud(df, cloud_type):
    plt.figure(figsize=(40, 30))
    text = ' '.join(list(df['text']))
    fd = FreqDist(text.split())
    word_cloud_text = ' '.join(
        [i[0] for i in fd.most_common(2000)])
    wordcloud = WordCloud(width=3000, height=2000, background_color='black',
                          colormap='seismic', collocations=False, stopwords=STOPWORDS).generate(word_cloud_text)
    wordcloud.to_file('./Figures/{}_word_cloud.png'.format(cloud_type))


def plot_word_clouds(df):
    reliable = df[df['label'] == 0]
    plot_word_cloud(reliable, 'reliable')
    unreliable = df[df['label'] == 1]
    plot_word_cloud(unreliable, 'unreliable')


def plot_confusion_matrix(cm, model_type, title):
    plt.figure(figsize=(12, 8))
    cm_df = pd.DataFrame(cm, index=['Unreliable', 'Reliable'], columns=[
                         'Unreliable', 'Reliable'])
    sns.heatmap(cm_df, annot=True, fmt='g')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('{} Confusion Matrix'.format(title))
    plt.savefig('./Figures/confusion_matrix_{}.png'.format(model_type))


def plot_roc_curve(fpr, tpr, model_type, title):
    plt.figure(figsize=(12, 8))
    roc_data = {
        'fpr': np.array(fpr),
        'tpr': np.array(tpr)
    }
    roc_baseline = {
        'fpr': np.array([0, 1]),
        'tpr': np.array([0, 1])
    }
    roc_data = pd.DataFrame(roc_data)
    roc_baseline = pd.DataFrame(roc_baseline)
    sns.lineplot('fpr', 'tpr', data=roc_data, label='Model ROC Curve')
    sns.lineplot('fpr', 'tpr', data=roc_baseline, label='Baseline ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} ROC Curve'.format(title))
    plt.legend()
    plt.savefig('./Figures/roc_curve_{}.png'.format(model_type))
