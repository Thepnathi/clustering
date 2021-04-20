import matplotlib.pyplot as plt
from typing import List

class Plot_B_CUBED:
    def plot_b_cubed(self, precision: List[float], recall: List[float], f_score: List[float], k: List[int], title="B-CUBED Measure",):
       plt.plot(k, precision, color="red", marker="X", label="Precision")
       plt.plot(k, recall, color="green", marker="o", label="Recall")
       plt.plot(k, f_score, color="blue", marker="^", label="F-score")
       plt.title(title, fontsize=14)
       plt.xlabel('F values', fontsize=10)
       plt.ylabel('B-CUBED precision, recall and F-score', fontsize=10)
       plt.legend(loc="upper left")
       plt.grid(True)
       plt.show()


if __name__ == "__main__":
    plot_tool = Plot_B_CUBED()
    plot_tool.plot_b_cubed([],[],[],[])

