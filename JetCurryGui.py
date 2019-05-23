import os
try:
    import tkinter as tk # python 3
except ImportError:
    import Tkinter as tk # python 2
from PIL import Image, ImageTk
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class JetCurryGui():
    '''
    GUI to display FITS image to select regions of interest for jet
    '''

    def __init__(self, file):
        self.gui = tk.Tk()
        self.gui.title('FITS Image')
        self.file = file

        # Frame to display the image
        self.master_frame = tk.Frame(self.gui, height=50, width=50)
        self.master_frame.grid(row=0, column=0)

        # Interactive frame to choose streams and display values
        self.i_frame = tk.Frame(self.gui, height=50, width=50)
        self.i_frame.grid(row=1, column=0)

        self.x_start_variable = tk.IntVar()
        self.y_start_variable = tk.IntVar()
        self.x_end_variable = tk.IntVar()
        self.y_end_variable = tk.IntVar()

        self.x_start_label = tk.Label(self.i_frame, text='X').grid(
            row=1, column=1)
        self.y_start_lable = tk.Label(self.i_frame, text='Y').grid(
            row=1, column=2)
        self.pixel_value_label = tk.Label(
            self.i_frame, text='Value').grid(
                row=1, column=3)

        self.start_point_label = tk.Label(
            self.i_frame, text='Start stream: ').grid(
                row=2, column=0)
        self.x_start_entry = tk.Entry(
            self.i_frame,
            textvariable=self.x_start_variable,
            width=5)
        self.x_start_entry.grid(row=2, column=1)
        self.y_start_entry = tk.Entry(
            self.i_frame,
            textvariable=self.y_start_variable,
            width=5)
        self.y_start_entry.grid(row=2, column=2)
        self.start_pixel_value_label = tk.Label(self.i_frame, text='')
        self.start_pixel_value_label.grid(row=2, column=3)

        self.end_point_label = tk.Label(
            self.i_frame, text='End stream: ').grid(
                row=3, column=0)
        self.x_end_entry = tk.Entry(
            self.i_frame,
            textvariable=self.x_end_variable,
            width=5)
        self.x_end_entry.grid(row=3, column=1)
        self.y_end_entry = tk.Entry(
            self.i_frame,
            textvariable=self.y_end_variable,
            width=5)
        self.y_end_entry.grid(row=3, column=2)
        self.end_pixel_value_label = tk.Label(self.i_frame, text='')
        self.end_pixel_value_label.grid(row=3, column=3)

        self.run_button = tk.Button(
            self.i_frame,
            text='Run',
            command=self.close_gui).grid(
                row=4,
                column=3)

        self.fits_data = fits.getdata(self.file)
        plt.imsave('dummy.jpg', self.fits_data)

        self.img = Image.open('dummy.jpg')
        self.photo = ImageTk.PhotoImage(self.img)
        self.panel = tk.Label(self.master_frame, image=self.photo)
        self.clicks = list(range(2))
        # Left mouse click for upstream (start)
        self.panel.bind('<Button-1>', self.get_start_point)
        # Right mouse click for downstream (end)
        self.panel.bind('<Button-2>', self.get_end_point)
        self.panel.grid(row=1, column=0)

        os.remove('dummy.jpg')

        self.gui.mainloop()

    def get_start_point(self, event):
        '''
        Get and display button clicks on image as upstream
        '''
        self.clicks[0:2] = event.x, event.y
        self.x_start_entry.delete(0, tk.END)
        self.x_start_entry.insert(0, self.clicks[0])
        self.y_start_entry.delete(0, tk.END)
        self.y_start_entry.insert(0, self.clicks[1])
        self.start_pixel_value_label.configure(
            text=self.fits_data[event.y - 1, event.x - 1])
        self.start_pixel_value_label.grid(row=2, column=3)

    def get_end_point(self, event):
        '''
        Get and display button clicks on image as downstream
        '''
        self.clicks[0:2] = event.x, event.y
        self.x_end_entry.delete(0, tk.END)
        self.x_end_entry.insert(0, self.clicks[0])
        self.y_end_entry.delete(0, tk.END)
        self.y_end_entry.insert(0, self.clicks[1])
        self.end_pixel_value_label.configure(
            text=self.fits_data[event.y - 1, event.x - 1])
        self.end_pixel_value_label.grid(row=3, column=3)

    def close_gui(self):
        '''
        Close GUI window
        Print warning if any input is empty
        '''
        if self.x_start_variable.get() != '' and self.y_start_variable.get(
        ) != '' and self.x_end_variable.get() != '' and self.y_end_variable.get() != '':
            self.gui.destroy()
        else:
            print('Missing an input!')


if __name__ == "__main__":
    GUI = JetCurryGui()
