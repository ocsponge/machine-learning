from numpy import *
from tkinter import *
import regtrees
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def redraw(tols, toln):
    redraw.f.clf()
    redraw.a = redraw.f.add_subplot(111)
    if chkbtnvar.get():
        if toln < 2:
            toln = 2
        mytree = regtrees.create_tree(redraw.rawdat, regtrees.model_leaf,
                                      regtrees.model_err, (tols, toln))
        yhat = regtrees.create_forecast(mytree, redraw.testdat,
                                        regtrees.model_tree_eval)
    else:
        mytree = regtrees.create_tree(redraw.rawdat, ops=(tols, toln))
        yhat = regtrees.create_forecast(mytree, redraw.testdat)
    redraw.a.scatter(redraw.rawdat[:, 0], redraw.rawdat[:, 1], s=5)
    redraw.a.plot(redraw.testdat, yhat, linewidth=2.0)
    redraw.canvas.show()


def get_input():
    try:
        toln = int(toln_entry.get())
    except:
        toln = 10
        print('enter integer for toln')
        toln_entry.delete(0, END)
        toln_entry.insert(0, '10')
    try:
        tols = float(tols_entry.get())
    except:
        tols = 1.0
        print('enter float for tols')
        tols_entry.delete(0, END)
        tols_entry.insert(0, '1.0')
    return tols, toln


def draw_new_tree():
    tols, toln = get_input()
    redraw(tols, toln)


root = Tk()

redraw.f = Figure(figsize=(5, 4), dpi=100)
redraw.canvas = FigureCanvasTkAgg(redraw.f, master=root)
redraw.canvas.show()
redraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

Label(root, text='toln').grid(row=1, column=0)
toln_entry = Entry(root)
toln_entry.grid(row=1, column=1)
toln_entry.insert(0, '10')
Label(root, text='tols').grid(row=2, column=0)
tols_entry = Entry(root)
tols_entry.grid(row=2, column=1)
tols_entry.insert(0, '1.0')
Button(root, text='redraw', command=draw_new_tree).grid(row=2, column=2)
chkbtnvar = IntVar()
chkbtn = Checkbutton(root, text='model tree', variable=chkbtnvar)
chkbtn.grid(row=3, columnspan=2)
redraw.rawdat = mat(regtrees.load_data_set('sine.txt'))
redraw.testdat = arange(
    min(redraw.rawdat[:, 0]), max(redraw.rawdat[:, 0]), 0.01)
redraw(1.0, 10)

root.mainloop()
