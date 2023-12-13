import h5py
import warnings
import numpy as np
import pynbody as pynbody
from astropy import units as u
from collections import Union
from astrochempy.utils import user_home
from astrochempy.gizmopy.utils import read_chem_info
from astrochempy.gizmopy.database import *

class SimGizmo:
    def __init__(self,
                 snapshot: str,
                 clump_model: str,
                 load_species: list) -> None:
        # Load snapshot and set unit system
        self.sim = self.__load_snapshot_ignoring_warning__(snapshot)

        # Set snapshot number and suffix strings
        file_name = ("snapshot" + snapshot.split("snapshot")[-1]).split(".")[0]
        self.number = file_name.split("_")[1]
        self.suffix = file_name.split("_")[2] if len (file_name.split("_")) > 2 else ""

        # Load time in Myr using h5py and convert it to kyr
        self.header = h5py.File(snapshot, "r")
        self.current_time = self.header["Header"].attrs["Time"] * 1e3

        # Rescale smoothing length to match Gasoline's definition of kernel extending up to 2*h
        self.sim.g['smooth'] /= 2

        # Load clump physical properties as a dictionary
        setup = globals()[clump_model]
        self.clump_props = {
            "r_BE" : setup.r_be * u.pc,
            "m_BE" : setup.m_be * u.Msun,
            "t_ff" : setup.t_ff * u.kyr,
            "size" : setup.size,
            "zoom" : setup.zoom,
            "avir" : setup.avir,
            "avgB" : setup.avgB,
            "Nmin" : setup.Nmin,
            "Nmax" : setup.Nmax
        }

        # Load chemistry data (species names, index and mass)
        self.species = None
        self.number_of_species = None
        self.load_chemistry_data()

        # Containers of all and gas particles data
        self.all = self.sim
        self.gas = self.sim.g

    def __load_snapshot_ignoring_warning__(self,
                                           snapshot: str) -> pynbody.SimSnap:
        """
        Load a snapshot ignoring the warning about missing units.
        
        Args:
            snapshot (str): The snapshot to load.

        Returns:
            pynbody.SimSnap: The loaded snapshot.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="No unit information found in GadgetHDF file. Using gadget default units."
            )

            sim = pynbody.load(snapshot)
            sim.set_units_system(velocity="km s^-1", mass="Msol", temperature="K", distance="pc")
            sim.physical_units(distance="pc")
            sim.g["rho"].convert_units("g cm^-3")

        return sim

    def close(self):
        """ Closes the h5py instance of the `header` to avoid memory leaks. """
        self.header.close()

    def get_clump_attributes(self,
                             column_size: int = 30) -> None:
        """ Prints the clump attributes. """
        print(f"\nClump attributes for snapshot {self.number}:")

        # map clump attributes keys to their physical names
        clump_keys = {
            "r_BE" : "Bonnor-Ebert radius",
            "m_BE" : "Bonnor-Ebert mass",
            "t_ff" : "free-fall time",
            "size" : "Map size",
            "zoom" : "Zoom factor",
            "avir" : "Virial parameter",
            "avgB" : "Average magnetic field",
            "Nmin" : "Minimum number column density",
            "Nmax" : "Maximum number column density"
        }

        # print clump attributes with their physical names
        for key, val in self.clump_props.items():
            print(f"{clump_keys[key]: <{column_size}}: {val}")

        # print header attributes
        print(f"\nSnapshot {self.number} has the following header attributes:")
        for key, val in self.header["Header"].attrs.items():
            print(f"{key: <{column_size}}: {val}")

    def gas_keys(self, column_size: int = 30):
        """ Prints main information about particle data of the simulation. """
        # Print gas particle data
        print('\nThe gas particles have the following data...')
        print(f'{"Keys": <{column_size}}{"Shape": <{column_size}}{"Units": <{column_size}}')
        for key in self.gas.loadable_keys():
            shape = str(self.gas[key].shape)
            units = str(self.gas[key].units)
            print(f'{key: <{column_size}}{shape: <{column_size}}{units: <{column_size}}')

    def load_chemistry_data(self):
        """ Load chemistry data (species names, index and mass) from the KROME simulation. """
        # Load species data from KROME
        species = read_chem_info(f"{user_home}/ammonia_paper/info.dat")
        self.species = species
        self.number_of_species = len(species)

        # Load mass fractions, convert them to number densities and store them in a new array
        for species in self.load_species:
            index, mass = self.species[species] # Species index and mass in g
            self.sim.g[species] = pynbody.array.SimArray(
                self.sim.g["KromeSpecies"][:, index - 1] * self.sim.g["rho"] / mass, "cm^-3"
            )

    def make_image(self,
                   field : str,
                   units : str,
                   norm1 : float = 1,
                   norm2 : float = 1,
                   resolution : int = 1500,
                   recenter : str | None = None,
                   width : str = None,
                   length : str = "0.2 pc",
                   orientation : str = "xy",
                   av_z : bool = False):
        """
        Makes a map of a given quantity `field`. The map can be projected or a slice dependending on `units` or `av_z`
        """
        # Adjust the image in accord to the required parameters of recentering, column length and orientation
        slab = self.Slab(recenter, orientation, length)

        if width is not None:
            _width = width
        else:
            _width = self.clump_props["size"]

        # Make image
        image = pynbody.plot.sph.image(slab.g, qty=field, units=units, width=_width, noplot=True,
                                       log=False, av_z=av_z, resolution=resolution) / norm1 / norm2

        # # Plot the image if plot is True
        # if plot:
        #     plot_map(image, self.clump_props["size"], self.zoom, vlim=vlim, log10=log10, showplot=showplot, clabel=clabel,
        #              cticks=cticks, xy_units=self.gas["pos"].units, time=self.current_time, cmap=cmap, beam=beam,
        #              filename=filename, show_ticks=show_ticks)

        return image

    def make_profile(self, field, resolution=1500, norm=1.0, weight="itself", filename=None, nbins=500):
        """
        Compute a radial profile of a given quantity.

        Args
        ----
        field: array-like
          The quantity for which the radial profile is computed.
        resolution: int or float, optional
          The resolution of the image. Default is 1500.
        norm: float, optional
          Normalization factor for the radial profile. Default is 1.0.
        weight: str or None, optional
          Weighting scheme for the radial profile. Default is 'itself'.
        filename: str or None, optional
          Name of the file to save the radial profile. Default is None.
        nbins: int, optional
          Number of bins for the radial profile. Default is 1500.

        Returns
        -------
        tuple:
          A tuple containing the radial positions in pc and the computed radial profile.
        """
        map_size = self.clump_props["size"] * 2
        qty_flat = field.flatten()
        rbins = np.logspace(-5, 0, nbins)
        rpos = (rbins[1:] + rbins[:-1]) * self.clump_props["size"] / 2
        pixpos = np.arange(0, resolution, 1)
        ypix, xpix = np.meshgrid(pixpos, pixpos)
        rpix = np.sqrt((xpix / resolution - 0.5)**2 + (ypix / resolution - 0.5)**2)

        if weight == "itself":
            num, _ = np.histogram(rpix.flatten(), bins=rbins, weights=qty_flat**2)
            den, _ = np.histogram(rpix.flatten(), bins=rbins, weights=qty_flat)
            field_prof = (num / den) / norm
        elif weight is None:
            field_hist, _ = np.histogram(rpix.flatten(), bins=rbins, weights=qty_flat)
            field_prof = field_hist * (map_size / resolution)**2 / (np.pi * (rbins[1:]**2 - rbins[:-1]**2) * map_size**2) / norm
        else:
            raise ValueError("External weights are not implemented yet -- please use either None or 'itself'.")

        if filename is not None:
            np.savetxt(f"{filename}.txt", np.transpose([rpos, field_prof]))

        return rpos, field_prof

    def make_3d_profile(self,
                        field,
                        norm : float = 1.0,
                        filename : str | None = None,
                        binning : tuple[str, float, float] = ("log", -5, 0),
                        nbins : int = 500):
        """
        Calculate the 3D radial profile of a given field around a central point.

        Args:
        -----
        field (array-like):
            Array of the same length as the gas particles containing the
            value of the field for each gas particle.
        norm (float, optional):
            Normalization factor for the radial profile. Default is 1.0.
        filename (str, optional):
            Name of the file to save the radial profile data.
            If None, data is not saved. Default is None.
        nbins (int, optional):
            Number of radial bins for the profile calculation.
            Default is 500.
        binning (tuple, optional):
            A tuple specifying the binning type and parameters.
            The tuple format is (str, min, max):
            - str: Either "log" for logarithmic space or "lin" for linear space.
            - min: The minimum radius value (if using linear space) or exponent
                      (if using logarithmic space).
            - max: The maximum radius value (if using linear space) or exponent
                      (if using logarithmic space).
            Default is ("log", -5, 0), i.e., from 1e-5 to 1 pc.
            
        Returns
        -------
        tuple:
            A tuple containing two arrays:
            - rpos (array): Radial positions corresponding to the center of each bin.
            - field_prof (array): Normalized 3D radial profile of the field.
        """
        # Constants for conversion
        pc2cm = 3.08567758e18 # Parsec to centimeter conversion factor
        Msun2g = 1.989e33     # Solar masses to grams conversion factor

        # Handling of the binning
        binning_type, min, max = binning

        if binning_type == "log":
            rbins = np.logspace(min, max, nbins)
        elif binning_type == "lin":
            rbins = np.linspace(min, max, nbins)
        else:
            raise ValueError("Invalid binning type. Use 'log' or 'lin'.")

        # Convert from parsecs to centimeters
        rpos = (rbins[1:] + rbins[:-1]) * self.clump_props["size"] / 2

        # Calculate the profile
        mass = self.gas["mass"] * Msun2g
        radius = self.gas["r"] * pc2cm
        weight = mass * self.gas[field]
        field_hist, _ = np.histogram(radius, rbins * pc2cm, weights=weight)
        volume_bins = ((rbins[1:]**3 - rbins[:-1]**3)  * (pc2cm**3)) * 4 * np.pi / 3
        field_prof = field_hist / (volume_bins) / norm

        # Save the profile on a text file if wanted
        if filename is not None:
            np.savetxt(f"{filename}.txt", np.transpose([rpos, field_prof]))

        return rpos, field_prof

    def make_species_profile(self,
                             species : str,
                             resolution : int = 1500,
                             nbins : int = 500,
                             filename : str | None = None,
                             ndim : int = 2,
                             weight = None,
                             relative : bool = True,
                             isomer : str | None = None) -> tuple[list, list]:
        try:
            # Try to get ortho and para maps
            oSp = self.make_image(f"{species}_ORTHO", "cm^-2", resolution=resolution, recenter="ssc")
            pSp = self.make_image(f"{species}_PARA" , "cm^-2", resolution=resolution, recenter="ssc")
            r, prof_ortho = self.make_profile(oSp, resolution=resolution, weight=weight, nbins=nbins, filename=filename, ndim=ndim)
            _, prof_para = self.make_profile(pSp, resolution=resolution, weight=weight, nbins=nbins, filename=filename, ndim=ndim)
        except KeyError:
            # If ortho and para maps not found, get total map
            tSp = self.make_image(species, "cm^-2", resolution=resolution, plot=False, recenter="ssc")
            r, prof_tSp = self.make_profile(tSp, resolution=resolution, weight=weight, nbins=nbins, filename=filename, ndim=ndim)
            prof_ortho = prof_tSp
            prof_para = None

        # Check if H2-relative or absolute column density profile
        if relative:
            oH2 = self.make_image("H2_ORTHO", "cm^-2", resolution=resolution, recenter="ssc")
            pH2 = self.make_image("H2_PARA", "cm^-2", resolution=resolution, recenter="ssc")
            _, prof_oH2 = self.make_profile(oH2, resolution=resolution, weight=weight, nbins=nbins, filename=filename, ndim=ndim)
            _, prof_pH2 = self.make_profile(pH2, resolution=resolution, weight=weight, nbins=nbins, filename=filename, ndim=ndim)
            norm_factor = prof_oH2 + prof_pH2
        else:
            norm_factor = 1

        # Determine which profile to plot based on isomer input
        if isomer is not None:
            isomer = isomer.lower()

            if isomer == "ortho":
                prof_qty = prof_ortho
            elif isomer == "para":
                prof_qty = prof_para
            elif isomer == "both":
                prof_qty = prof_para + prof_ortho if prof_para is not None else prof_ortho
            else:
                raise ValueError(f"Invalid value `{isomer}` for `isomer`. Must be `ortho`, `para`, or `both`.")
        else:
            prof_qty = prof_ortho

        # Define the quantity to be plotted
        profile = prof_qty / norm_factor

        return r, profile

    class Subview:
        def __init__(self,
                     sub_filter: tuple,
                     recenter_method: Union[str, None] = None) -> None:
            self.filter, self.filter_props = sub_filter
            self.recenter_method = recenter_method

        def _recenter(self):
            _map_recenter_method = {
                "com" : "center_of_mass",
                "ssc" : "shrinking_sphere",
                "pot" : "center_of_potential"
            }

            # recenter the simulation
            if self.recenter_method is None:
                for axis in "xyz":
                    self.sim.g[axis] -= np.average(self.sim.g[axis], weights=self.sim.g["rho"])
            elif self.recenter_method.lower() == "maximum_rho":
                new_pos = np.argmax(self.sim.g["rho"])
                self.sim["pos"] -= self.sim.g["pos"][new_pos]
            elif self.recenter_method.lower() in _map_recenter_method.keys():
                new_pos = pynbody.analysis.halo.center(
                    self.sim,
                    mode=_map_recenter_method[self.recenter_method.lower()],
                    retcen=True
                )
            else:
                raise ValueError("Invalid recenter method. Use 'ssc', 'maximum_rho', 'center_of_mass', or 'center_of_potential'.")

        def _slab(self,
                  orientation: str = "xy"):
            """ Return particles in a slab of thickness `height` orientated in the `orientation` plane. """
            # Define a column of length `height` in the desired unit
            height_value, height_unit = self.filter_props.split(" ")
            column = f"{float(height_value)} {height_unit}"

            # Define the slab with its orientation and length
            if orientation.lower() == "xy":
                slab = self.all[pynbody.filt.BandPass("z", f"-{column}", column)]
            elif orientation.lower() == "xz":
                self.all.rotate_x(90)
                slab = self.all[pynbody.filt.BandPass("y", f"-{column}", column)]
            elif orientation.lower() == "yz":
                self.all.rotate_y(90)
                slab = self.all[pynbody.filt.BandPass("x", f"-{column}", column)]
            else:
                raise ValueError("Invalid orientation. Use 'xy', 'xz', or 'yz'.")
            
            return slab
        
        def _sphere(self,
                    radius: str = "0.1 pc",
                    center: str = (0, 0, 0)):
            """ Return particles within `radius` of the point `centre`. """
            # Define the sphere with its radius
            sphere = self.all[pynbody.filt.Sphere(self.filter_props[0], cen=self.filter_props[1])]

            return sphere
