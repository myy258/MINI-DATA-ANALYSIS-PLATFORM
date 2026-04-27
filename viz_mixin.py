import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from PyQt5.QtWidgets import QMessageBox

class VizMixin:
    def update_viz_controls(self):
        viz_type = self.viz_type_combo.currentText()
        disabled = '预测vs实际' in viz_type
        self.viz_x_combo.setEnabled(not disabled)
        self.viz_y_combo.setEnabled(not disabled)

    def generate_plot(self):
        if self.df is None:
            QMessageBox.warning(self, '警告', '请先加载数据')
            return
        try:
            viz_type = self.viz_type_combo.currentText()
            self.figure.clear()
            if '预测vs实际' in viz_type:
                self.plot_prediction_vs_actual()
            elif '多变量对比' in viz_type:
                self.plot_multi_variable()
            elif '散点图+拟合线' in viz_type:
                self.plot_scatter_with_fit()
            elif '直方图' in viz_type:
                self.plot_histogram()
            elif '箱线图' in viz_type:
                self.plot_boxplot()
            elif '热力图' in viz_type:
                self.plot_heatmap()
            self.figure.tight_layout(pad=2.5)
            self.canvas.draw()
            self.tabs.setCurrentWidget(self.viz_tab)
        except Exception as e:
            QMessageBox.critical(self, '错误', f'生成图表失败:\n{str(e)}')

    # ── UPDATED: shows training set (left) + test set (right) ──────────────
    def plot_prediction_vs_actual(self):
        if self.last_predictions is None or self.last_actual is None:
            QMessageBox.warning(self, '警告', '请先训练一个模型再查看预测结果')
            return

        has_train = (self.last_train_predictions is not None
                     and self.last_train_actual is not None)

        if has_train:
            ax_train = self.figure.add_subplot(1, 2, 1)
            ax_test  = self.figure.add_subplot(1, 2, 2)
        else:
            ax_test  = self.figure.add_subplot(1, 1, 1)

        def _draw_panel(ax, actual, predicted, title, color):
            ax.scatter(actual, predicted, alpha=0.55, s=50,
                       color=color, edgecolors='black', linewidths=0.4, label='数据点')
            min_v = min(actual.min(), predicted.min())
            max_v = max(actual.max(), predicted.max())
            ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='理想线 (y=x)')
            r2 = r2_score(actual, predicted)
            ax.text(0.95, 0.05, f'R² = {r2:.4f}', transform=ax.transAxes,
                    fontsize=12, va='bottom', ha='right', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffe69c',
                              alpha=0.9, edgecolor='#5a6c7d', linewidth=1.5))
            ax.set_xlabel('实际值',  fontsize=12, fontweight='bold')
            ax.set_ylabel('预测值',  fontsize=12, fontweight='bold')
            ax.set_title(title,      fontsize=13, fontweight='bold', pad=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='upper left', framealpha=0.9)

        # Test set panel
        _draw_panel(ax_test,
                    np.asarray(self.last_actual),
                    np.asarray(self.last_predictions),
                    '测试集: 预测值 vs 实际值',
                    '#e07b5a')

        # Training set panel (only when data available)
        if has_train:
            train_actual = np.asarray(self.last_train_actual)
            train_preds  = np.asarray(self.last_train_predictions)
            _draw_panel(ax_train, train_actual, train_preds,
                        '训练集: 预测值 vs 实际值', '#6b9fd4')
    # ── END UPDATED ─────────────────────────────────────────────────────────

    def plot_histogram(self):
        x_cols = self.viz_x_combo.get_checked_items()
        if not x_cols:
            QMessageBox.warning(self, '警告', '请选择X轴变量')
            return
        n_plots = len(x_cols)
        if n_plots == 1:      rows, cols = 1, 1
        elif n_plots == 2:    rows, cols = 1, 2
        elif n_plots <= 4:    rows, cols = 2, 2
        elif n_plots <= 6:    rows, cols = 2, 3
        elif n_plots <= 9:    rows, cols = 3, 3
        else:                 rows, cols = 4, 3
        for idx, x_col in enumerate(x_cols[:12]):
            ax   = self.figure.add_subplot(rows, cols, idx + 1)
            data = self.df[x_col].dropna()
            if pd.api.types.is_numeric_dtype(data):
                ax.hist(data, bins=30, edgecolor='black', color='#6b7d8e', alpha=0.7)
            else:
                vc = data.value_counts()
                ax.bar(range(len(vc)), vc.values, color='#6b7d8e', alpha=0.7)
                ax.set_xticks(range(len(vc)))
                ax.set_xticklabels(vc.index, rotation=45, ha='right')
            ax.set_title(x_col, fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel(x_col, fontsize=10)
            ax.set_ylabel('频数', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)

    def plot_boxplot(self):
        x_cols = self.viz_x_combo.get_checked_items()
        if not x_cols:
            QMessageBox.warning(self, '警告', '请选择变量')
            return
        ax = self.figure.add_subplot(111)
        data_to_plot, labels = [], []
        for x_col in x_cols:
            col_data = self.df[x_col].dropna()
            if pd.api.types.is_numeric_dtype(col_data):
                data_to_plot.append(col_data)
                labels.append(x_col)
        if data_to_plot:
            ax.boxplot(data_to_plot, labels=labels)
            ax.set_title('箱线图对比', fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel('值', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            QMessageBox.warning(self, '警告', '没有数值型数据可绘制箱线图')

    def plot_heatmap(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            QMessageBox.warning(self, '警告', '需要至少2个数值型列生成热力图')
            return
        ax   = self.figure.add_subplot(111)
        corr = self.df[numeric_cols].corr()
        im   = ax.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(corr.columns, fontsize=10)
        ax.set_title('相关性热力图', fontsize=14, fontweight='bold', pad=20)
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=8)
        cbar = self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)

    def plot_scatter_with_fit(self):
        x_cols = self.viz_x_combo.get_checked_items()
        y_cols = self.viz_y_combo.get_checked_items()
        if not x_cols or not y_cols:
            QMessageBox.warning(self, '警告', '请选择X轴和Y轴变量')
            return
        combinations = [(x, y) for x in x_cols for y in y_cols]
        n_plots      = len(combinations)
        if n_plots == 1:      rows, cols = 1, 1
        elif n_plots == 2:    rows, cols = 1, 2
        elif n_plots <= 4:    rows, cols = 2, 2
        elif n_plots <= 6:    rows, cols = 2, 3
        elif n_plots <= 9:    rows, cols = 3, 3
        else:                 rows, cols = 4, 3
        for idx, (x_col, y_col) in enumerate(combinations[:12]):
            ax   = self.figure.add_subplot(rows, cols, idx + 1)
            data = self.df[[x_col, y_col]].dropna()
            if len(data) < 2:
                continue
            x, y = data[x_col].values, data[y_col].values
            ax.scatter(x, y, alpha=0.6, s=40, color='#6b7d8e', edgecolors='black', linewidths=0.5)
            coeffs   = np.polyfit(x, y, 1)
            poly_f   = np.poly1d(coeffs)
            x_sorted = np.sort(x)
            ax.plot(x_sorted, poly_f(x_sorted), 'r-', linewidth=2, alpha=0.8)
            r2 = r2_score(y, poly_f(x))
            ax.text(0.05, 0.95, f'R²={r2:.3f}', transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.85))
            ax.set_title(f'{x_col} vs {y_col}', fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel(x_col, fontsize=9)
            ax.set_ylabel(y_col, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

    def plot_multi_variable(self):
        x_cols = self.viz_x_combo.get_checked_items()
        y_cols = self.viz_y_combo.get_checked_items()
        if not x_cols or not y_cols:
            QMessageBox.warning(self, '警告', '请选择X轴和Y轴变量')
            return
        x_col   = x_cols[0]
        ax      = self.figure.add_subplot(111)
        colors  = ['#6b7d8e', '#8b9dad', '#5a6c7d', '#7a8c9d', '#4a5c6d', '#9aacbd']
        markers = ['o', 's', '^', 'D', 'v', '<']
        for i, y_col in enumerate(y_cols):
            data = self.df[[x_col, y_col]].dropna()
            if len(data) > 0:
                ax.plot(data[x_col], data[y_col],
                        marker=markers[i % len(markers)], label=y_col,
                        linewidth=2, markersize=6,
                        color=colors[i % len(colors)], alpha=0.7)
        ax.set_title('多变量对比', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel('值', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10, framealpha=0.9)