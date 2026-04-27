import numpy as np
import pandas as pd
from scipy import stats
from PyQt5.QtWidgets import QMessageBox

class StatsMixin:
    def show_descriptive_stats(self):
        if self.df is None:
            QMessageBox.warning(self, '警告', '请先加载数据')
            return
        try:
            self._show_stats_text()
            result = "=" * 80 + "\n描述性统计分析\n" + "=" * 80 + "\n\n"
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                result += "【数值型变量统计】\n" + "-" * 80 + "\n"
                result += self.df[numeric_cols].describe().to_string() + "\n\n"
            cat_cols = self.df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                result += "【分类型变量统计】\n" + "-" * 80 + "\n"
                for col in cat_cols:
                    result += f"\n{col}:\n"
                    result += self.df[col].value_counts().to_string()
                    result += f"\n  唯一值数量: {self.df[col].nunique()}\n"
            self.stats_result.setText(result)
            self.tabs.setCurrentWidget(self.stats_tab)
        except Exception as e:
            QMessageBox.critical(self, '错误', f'统计分析失败:\n{str(e)}')

    def show_correlation(self):
        if self.df is None:
            QMessageBox.warning(self, '警告', '请先加载数据')
            return
        try:
            self._show_stats_text()
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                QMessageBox.warning(self, '警告', '需要至少2个数值型列进行相关性分析')
                return
            corr_matrix = self.df[numeric_cols].corr()
            result = "=" * 80 + "\n相关性分析 (Pearson相关系数)\n" + "=" * 80 + "\n\n"
            result += corr_matrix.to_string() + "\n\n"
            result += "【强相关变量对 (|r| > 0.7)】\n" + "-" * 80 + "\n"
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            if strong_corr:
                for var1, var2, corr_val in sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True):
                    result += f"{var1} <-> {var2}: {corr_val:.4f}\n"
            else:
                result += "未发现强相关变量对\n"
            self.stats_result.setText(result)
            self.tabs.setCurrentWidget(self.stats_tab)
        except Exception as e:
            QMessageBox.critical(self, '错误', f'相关性分析失败:\n{str(e)}')

    def test_normality(self):
        if self.df is None:
            QMessageBox.warning(self, '警告', '请先加载数据')
            return
        try:
            self._show_stats_text()
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                QMessageBox.warning(self, '警告', '没有数值型列可供检验')
                return
            result = ("=" * 80 + "\n正态分布检验 (Shapiro-Wilk Test)\n" + "=" * 80 + "\n"
                      "H0: 数据服从正态分布\n显著性水平: α = 0.05\n"
                      "判断: p-value > 0.05 则接受H0\n" + "-" * 80 + "\n\n")
            for col in numeric_cols:
                data = self.df[col].dropna()
                if len(data) > 3:
                    statistic, p_value = stats.shapiro(data)
                    result += f"{col}:\n  统计量: {statistic:.4f}\n  p-value: {p_value:.4f}\n"
                    result += ("  结论: ✓ 数据服从正态分布 (p > 0.05)\n\n" if p_value > 0.05
                               else "  结论: ✗ 数据不服从正态分布 (p ≤ 0.05)\n\n")
                else:
                    result += f"{col}: 数据量不足 (需要 > 3)\n\n"
            self.stats_result.setText(result)
            self.tabs.setCurrentWidget(self.stats_tab)
        except Exception as e:
            QMessageBox.critical(self, '错误', f'正态性检验失败:\n{str(e)}')

    def detect_outliers(self):
        if self.df is None:
            QMessageBox.warning(self, '警告', '请先加载数据')
            return
        try:
            self._show_stats_text()
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                QMessageBox.warning(self, '警告', '没有数值型列可供检测')
                return
            result = ("=" * 80 + "\n异常值检测 (IQR方法)\n" + "=" * 80 + "\n"
                      "方法: 四分位距 (Interquartile Range)\n"
                      "异常值定义: 值 < Q1 - 1.5×IQR 或 值 > Q3 + 1.5×IQR\n"
                      + "-" * 80 + "\n\n")
            total_outliers = 0
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                result += (f"{col}:\n  Q1 (25%): {Q1:.2f}\n  Q3 (75%): {Q3:.2f}\n"
                           f"  IQR: {IQR:.2f}\n  正常范围: [{lower_bound:.2f}, {upper_bound:.2f}]\n"
                           f"  异常值数量: {len(outliers)} ({len(outliers)/len(self.df)*100:.2f}%)\n")
                if len(outliers) > 0:
                    result += f"  示例异常值: {outliers[col].tolist()[:10]}\n"
                    total_outliers += len(outliers)
                result += "\n"
            result += f"\n总计检测到 {total_outliers} 个异常值\n"
            self.stats_result.setText(result)
            self.tabs.setCurrentWidget(self.stats_tab)
        except Exception as e:
            QMessageBox.critical(self, '错误', f'异常值检测失败:\n{str(e)}')

    def show_distribution_analysis(self):
        if self.df is None:
            QMessageBox.warning(self, '警告', '请先加载数据')
            return
        try:
            from scipy.stats import gaussian_kde, norm as sp_norm
    
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                QMessageBox.warning(self, '警告', '没有数值型列可供分析')
                return
    
            self.stats_figure.clear()
    
            n = len(numeric_cols)
            if   n == 1:     rows, cols = 1, 1
            elif n == 2:     rows, cols = 1, 2
            elif n <= 4:     rows, cols = 2, 2
            elif n <= 6:     rows, cols = 2, 3
            elif n <= 9:     rows, cols = 3, 3
            elif n <= 12:    rows, cols = 3, 4
            else:            rows, cols = 4, 4
    
            max_plots = rows * cols
    
            for idx, col in enumerate(numeric_cols[:max_plots]):
                ax   = self.stats_figure.add_subplot(rows, cols, idx + 1)
                data = self.df[col].dropna()
                if len(data) < 3:
                    ax.set_title(col + '\n(数据不足)', fontsize=10)
                    continue
    
                n_bins = min(30, max(10, len(data) // 10))
    
                # 柱状图（密度归一化）
                ax.hist(data, bins=n_bins,
                        color='#6b9fd4', alpha=0.65,
                        edgecolor='white', linewidth=0.5,
                        density=True, label='频率分布')
    
                x_range = np.linspace(data.min(), data.max(), 300)
    
                # KDE 曲线
                try:
                    kde = gaussian_kde(data)
                    ax.plot(x_range, kde(x_range),
                            color='#e07b5a', linewidth=2.2, label='KDE')
                except Exception:
                    pass
    
                # 正态参考曲线
                mu, sigma = data.mean(), data.std()
                if sigma > 0:
                    ax.plot(x_range, sp_norm.pdf(x_range, mu, sigma),
                            'g--', linewidth=1.5, alpha=0.75, label='正态参考')
    
                # Shapiro-Wilk / KS 检验结果标注
                if len(data) <= 5000:
                    _, p_value = stats.shapiro(data)
                else:
                    _, p_value = stats.kstest(data, 'norm', args=(mu, sigma))
    
                is_normal  = p_value > 0.05
                tag_text   = f'p={p_value:.3f}\n{"✓ 正态" if is_normal else "✗ 非正态"}'
                tag_color  = '#d4edda' if is_normal else '#fde8e8'
    
                ax.text(0.97, 0.97, tag_text,
                        transform=ax.transAxes, fontsize=8,
                        va='top', ha='right',
                        bbox=dict(boxstyle='round,pad=0.35',
                                  facecolor=tag_color, alpha=0.9, edgecolor='gray'))
    
                ax.set_title(col, fontsize=10, fontweight='bold', pad=6)
                ax.set_xlabel('值',  fontsize=9)
                ax.set_ylabel('密度', fontsize=9)
                ax.tick_params(labelsize=8)
                ax.grid(True, alpha=0.25)
                ax.legend(fontsize=7, loc='upper left', framealpha=0.8)
    
            self.stats_figure.suptitle('字段分布分析',
                                       fontsize=13, fontweight='bold')
            self.stats_figure.tight_layout(rect=[0, 0, 1, 0.97])
            self.stats_canvas.draw()
    
            self._show_stats_chart()
            self.tabs.setCurrentWidget(self.stats_tab)
    
        except Exception as e:
            QMessageBox.critical(self, '错误', f'分布分析失败:\n{str(e)}')
    
    def _show_stats_text(self):
        """切换到文字结果视图"""
        self.stats_result.show()
        self.stats_canvas.hide()

    def _show_stats_chart(self):
        """切换到图表视图"""
        self.stats_result.hide()
        self.stats_canvas.show()