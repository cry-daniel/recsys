# -*- coding: utf-8 -*-
"""
按日期分组分析：04.08 - 04.21 每天的 Average Sequence Length by Video ID
并追踪同一个 video_id 在不同天之间的 average_sequence_length 变化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data(data_path: str, nrows: int = None) -> pd.DataFrame:
    """加载数据"""
    print(f"正在加载数据: {data_path}")
    df = pd.read_csv(data_path, nrows=nrows)
    print(f"数据加载完成，共 {len(df)} 行")
    return df


def calculate_daily_avg_sequence_length(df: pd.DataFrame) -> dict:
    """
    按日期分组，计算每天的 video_id 平均序列长度
    
    返回: {date: {video_id: avg_sequence_length}}
    """
    results = {}
    dates = sorted(df['date'].unique())
    
    for date in dates:
        df_day = df[df['date'] == date]
        
        # 统计每个 user_id 的序列长度
        user_sequence_lengths = df_day.groupby('user_id').size()
        
        # 为每个 video_id 计算其出现的用户序列长度的平均值
        video_id_avg_seq = {}
        for video_id, group in df_day.groupby('video_id'):
            user_seq_lengths = user_sequence_lengths.loc[group['user_id'].unique()]
            video_id_avg_seq[video_id] = np.mean(user_seq_lengths)
        
        results[date] = video_id_avg_seq
        print(f"日期 {date}: 共 {len(video_id_avg_seq)} 个 video_id")
    
    return results


def get_high_frequency_videos(daily_results: dict, min_days: int = 10) -> list:
    """
    获取高频 video_id（在至少 min_days 天中出现的 video_id）
    """
    video_appearances = {}
    for date, video_dict in daily_results.items():
        for video_id in video_dict.keys():
            if video_id not in video_appearances:
                video_appearances[video_id] = 0
            video_appearances[video_id] += 1
    
    high_freq_videos = [v for v, count in video_appearances.items() if count >= min_days]
    print(f"在至少 {min_days} 天中出现的 video_id 数量: {len(high_freq_videos)}")
    return high_freq_videos


def get_top_videos_by_frequency(df: pd.DataFrame, top_n: int = 50) -> list:
    """获取出现次数最多的 Top N video_id"""
    video_counts = df['video_id'].value_counts()
    top_videos = video_counts.head(top_n).index.tolist()
    return top_videos


def plot_daily_distribution_subplots(daily_results: dict, output_dir: Path):
    """
    方案A：多子图展示
    每天一张子图，展示当天所有 video_id 按平均序列长度排序后的分布
    """
    dates = sorted(daily_results.keys())
    n_dates = len(dates)
    
    fig, axes = plt.subplots(n_dates, 1, figsize=(12, 3 * n_dates))
    if n_dates == 1:
        axes = [axes]
    
    for idx, date in enumerate(dates):
        video_avg_seq = daily_results[date]
        sorted_values = sorted(video_avg_seq.values())
        axes[idx].plot(sorted_values, linewidth=0.5)
        axes[idx].set_title(f'Date: {date}')
        axes[idx].set_xlabel('Video ID (Sorted)')
        axes[idx].set_ylabel('Avg Sequence Length')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'daily_distribution_subplots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_dir / 'daily_distribution_subplots.png'}")


def plot_heatmap(daily_results: dict, high_freq_videos: list, output_dir: Path, max_videos: int = 100):
    """
    方案B：热力图
    横轴：日期，纵轴：video_id，颜色：平均序列长度
    """
    # 限制 video_id 数量
    videos_to_plot = high_freq_videos[:max_videos]
    dates = sorted(daily_results.keys())
    
    # 构建矩阵
    matrix = np.zeros((len(videos_to_plot), len(dates)))
    for j, date in enumerate(dates):
        for i, video_id in enumerate(videos_to_plot):
            matrix[i, j] = daily_results[date].get(video_id, np.nan)
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(14, max(8, len(videos_to_plot) * 0.1)))
    sns.heatmap(matrix, 
                xticklabels=dates, 
                yticklabels=[f'v{i}' for i in range(len(videos_to_plot))],
                cmap='YlOrRd', 
                ax=ax,
                cbar_kws={'label': 'Average Sequence Length'})
    ax.set_xlabel('Date')
    ax.set_ylabel('Video ID')
    ax.set_title(f'Average Sequence Length Heatmap (Top {len(videos_to_plot)} Videos)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_dir / 'heatmap.png'}")


def plot_trend_lines(daily_results: dict, videos_to_track: list, output_dir: Path):
    """
    方案C：趋势折线图
    展示选定 video_id 的平均序列长度随时间的变化趋势
    """
    dates = sorted(daily_results.keys())
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(videos_to_track)))
    
    for idx, video_id in enumerate(videos_to_track):
        values = [daily_results[date].get(video_id, np.nan) for date in dates]
        ax.plot(dates, values, marker='o', markersize=4, 
                label=f'Video {video_id}', color=colors[idx], linewidth=1.5)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Sequence Length')
    ax.set_title('Average Sequence Length Trend by Video ID')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trend_lines.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_dir / 'trend_lines.png'}")


def plot_change_analysis(daily_results: dict, high_freq_videos: list, output_dir: Path):
    """
    方案D：变化量分析
    分析同一 video_id 在相邻天数之间的平均序列长度变化
    """
    dates = sorted(daily_results.keys())
    
    # 计算每个 video_id 的日间变化
    changes_by_video = {}
    for video_id in high_freq_videos:
        changes = []
        for i in range(1, len(dates)):
            prev_val = daily_results[dates[i-1]].get(video_id, np.nan)
            curr_val = daily_results[dates[i]].get(video_id, np.nan)
            if not (np.isnan(prev_val) or np.isnan(curr_val)):
                changes.append(curr_val - prev_val)
        if changes:
            changes_by_video[video_id] = changes
    
    # 绘制箱线图展示变化分布
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：所有 video_id 的变化分布（小提琴图）
    all_changes = []
    for video_id, changes in changes_by_video.items():
        for change in changes:
            all_changes.append({'video_id': video_id, 'change': change})
    
    if all_changes:
        df_changes = pd.DataFrame(all_changes)
        # 随机采样部分数据用于可视化
        sample_videos = np.random.choice(list(changes_by_video.keys()), 
                                         min(30, len(changes_by_video)), replace=False)
        df_sample = df_changes[df_changes['video_id'].isin(sample_videos)]
        
        sns.violinplot(data=df_sample, x='video_id', y='change', ax=axes[0])
        axes[0].set_xlabel('Video ID')
        axes[0].set_ylabel('Daily Change in Avg Sequence Length')
        axes[0].set_title('Distribution of Daily Changes (Sample of 30 Videos)')
        axes[0].tick_params(axis='x', rotation=90)
    
    # 右图：整体变化趋势
    mean_changes = []
    for i in range(1, len(dates)):
        day_changes = []
        for video_id in high_freq_videos:
            prev_val = daily_results[dates[i-1]].get(video_id, np.nan)
            curr_val = daily_results[dates[i]].get(video_id, np.nan)
            if not (np.isnan(prev_val) or np.isnan(curr_val)):
                day_changes.append(curr_val - prev_val)
        mean_changes.append(np.mean(day_changes) if day_changes else 0)
    
    axes[1].bar(range(len(mean_changes)), mean_changes, color='steelblue')
    axes[1].set_xlabel('Day Transition')
    axes[1].set_ylabel('Mean Change in Avg Sequence Length')
    axes[1].set_title('Mean Daily Change Across All Videos')
    axes[1].set_xticks(range(len(mean_changes)))
    axes[1].set_xticklabels([f'{dates[i]}->{dates[i+1]}' for i in range(len(mean_changes))], 
                            rotation=45, ha='right')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'change_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_dir / 'change_analysis.png'}")


def plot_summary_statistics(daily_results: dict, output_dir: Path):
    """
    绘制每日统计摘要
    """
    dates = sorted(daily_results.keys())
    
    stats = {
        'date': [],
        'num_videos': [],
        'mean_avg_seq_len': [],
        'median_avg_seq_len': [],
        'std_avg_seq_len': [],
        'max_avg_seq_len': [],
        'min_avg_seq_len': []
    }
    
    for date in dates:
        values = list(daily_results[date].values())
        stats['date'].append(date)
        stats['num_videos'].append(len(values))
        stats['mean_avg_seq_len'].append(np.mean(values))
        stats['median_avg_seq_len'].append(np.median(values))
        stats['std_avg_seq_len'].append(np.std(values))
        stats['max_avg_seq_len'].append(np.max(values))
        stats['min_avg_seq_len'].append(np.min(values))
    
    df_stats = pd.DataFrame(stats)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 每日 video_id 数量
    axes[0, 0].bar(dates, stats['num_videos'], color='steelblue')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Number of Videos')
    axes[0, 0].set_title('Daily Number of Unique Videos')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 每日平均序列长度的均值
    axes[0, 1].plot(dates, stats['mean_avg_seq_len'], marker='o', color='green', label='Mean')
    axes[0, 1].fill_between(dates, 
                            np.array(stats['mean_avg_seq_len']) - np.array(stats['std_avg_seq_len']),
                            np.array(stats['mean_avg_seq_len']) + np.array(stats['std_avg_seq_len']),
                            alpha=0.3, color='green')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Average Sequence Length')
    axes[0, 1].set_title('Daily Mean Avg Sequence Length (with Std)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].legend()
    
    # 每日中位数
    axes[1, 0].plot(dates, stats['median_avg_seq_len'], marker='s', color='orange', label='Median')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Average Sequence Length')
    axes[1, 0].set_title('Daily Median Avg Sequence Length')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].legend()
    
    # 每日最大最小值
    axes[1, 1].plot(dates, stats['max_avg_seq_len'], marker='^', color='red', label='Max')
    axes[1, 1].plot(dates, stats['min_avg_seq_len'], marker='v', color='blue', label='Min')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Average Sequence Length')
    axes[1, 1].set_title('Daily Max/Min Avg Sequence Length')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_dir / 'summary_statistics.png'}")
    
    return df_stats


def main():
    """主函数"""
    # 配置
    data_path = "tmp_data/KuaiRand-1K/data/log_standard_4_08_to_4_21_1k.csv"
    output_dir = Path("figs/daily_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    df = load_data(data_path, nrows=None)  # 设置 nrows=None 加载全部数据，或设置数字限制行数
    
    # 检查日期列
    print(f"\n日期范围: {df['date'].min()} 到 {df['date'].max()}")
    print(f"共有 {df['date'].nunique()} 个不同的日期")
    print(f"日期列表: {sorted(df['date'].unique())}")
    
    # 计算每天的 video_id 平均序列长度
    print("\n计算每天的 video_id 平均序列长度...")
    daily_results = calculate_daily_avg_sequence_length(df)
    
    # 获取高频 video_id
    high_freq_videos = get_high_frequency_videos(daily_results, min_days=10)
    
    # 获取 Top 50 高频 video_id（按出现次数）
    top_videos = get_top_videos_by_frequency(df, top_n=30)
    
    # 绘制各种图表
    print("\n开始生成可视化图表...")
    
    # 方案A：多子图展示
    plot_daily_distribution_subplots(daily_results, output_dir)
    
    # 方案B：热力图
    plot_heatmap(daily_results, high_freq_videos, output_dir, max_videos=100)
    
    # 方案C：趋势折线图
    plot_trend_lines(daily_results, top_videos[:20], output_dir)
    
    # 方案D：变化量分析
    plot_change_analysis(daily_results, high_freq_videos, output_dir)
    
    # 统计摘要
    df_stats = plot_summary_statistics(daily_results, output_dir)
    
    # 保存统计数据
    df_stats.to_csv(output_dir / 'daily_statistics.csv', index=False)
    print(f"\n统计数据已保存: {output_dir / 'daily_statistics.csv'}")
    
    print("\n分析完成！")


if __name__ == "__main__":
    main()