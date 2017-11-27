plt.hist(angles,bins=50,normed=True)
plt.xlabel("Angle (degree)")
plt.rc("text",usetex=False)
plt.rc("font",family="sans-serif",size=20)
plt.rcParams["figure.figsize"]=8,8
plt.rcParams["font.sans-serif"] = "Arial"
plt.ylabel("Frequency (normalized)")
plt.xticks(np.arange(30,151,20))
plt.savefig("plots/angle_hist.pdf",format="pdf",bbox_inches="tight")
# plt.title("Histogram Angle between the direction of gradients and the first right singular vector")