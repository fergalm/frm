# -*- coding: utf-8 -*-

"""
Read a list of geometries from a shapefile or kml file

Based on similar code by Alex

"""

import osgeo.ogr as ogr
import pandas as pd
import collections
import tempfile
import zipfile
import tarfile
import shutil
import re
import os


RE_SHAPEFILE = re.compile('.*\.(shp|kml|kmz|gml)$', re.IGNORECASE)
# File name extensions of recognized archive types
ARCHIVE_EXTENSIONS = set(['tar.gz', 'zip', 'gz', 'tar', 'tbz', 'tgz', 'bz2', 'bz'])


def load_geoms_as_dict(fn, idkey='Name'):
    """Load a kml file and store the geometries in a dictionary

    This function throws away any metadata stored with the geometry except for the Name
    So it's not always what you want to do.

    Optional Arguments
    --------------------
    idkey
        (string) The identifiying attribute of the shape to use a dictionary key
    """
    props = get_geometries(fn)

    out = dict()
    for pr in props:
        key = pr[1][idkey]
        value = pr[0]

        if key in out:
            raise KeyError("Duplicate names (%s) in %s" %(key, fn))
        out[key] = value
    return out


def load_geoms_as_df(fn, idkey=None, all_keys=True):
    """Load a kml as a dataframe

    Inputs
    fn
        (str) Name of file to read
    idkey
        (str) Which parameter to use an index
    all_keys
        (bool) If True, include all parameters in the dataframe
    Todo: Keep the metadata, dont' throw it away

    """

    if idkey is None and not all_keys:
        raise ValueError("Must specify idkey or take all keys")

    props = get_geometries(fn)

    if all_keys:
        keys = list(props[0][1].keys())
    else:
        keys = [idkey]

    df = pd.DataFrame()
    df['geoms'] = list(map(lambda x: x[0], props))

    for k in keys:
        df[k] = list(map(lambda x: x[1][k], props))

    if idkey is not None:
        df.index = df[idkey]
    return df


def get_geometries(*shapefile_or_archive, **kwargs):
    """
    Loads the geometry data from a list of shapefiles or archives containing a
    bunch of shapefiles each.

    Returns a list of tuples:

        [ (geometry, {attr: value, attr: value, ...}), ... ]

    The 'attr:value' dictionary are the attributes for each geometry.

    If you specify a 'get_metadata=False' argument, the return data will be
    a simple list of geometries -- no metadata.

    If you specify the 'strict=True' argument, any problem with the data (even
    minor) will result in an exceptions being thrown, instead of a warning.
    """
    geometries = []
    is_shapefile = lambda file_name: bool(RE_SHAPEFILE.match(file_name))
    for file_name in shapefile_or_archive:
        if is_archive(file_name):
            tempdir = extract_all(file_name)
            try:
                shapefiles = find(tempdir, RE_SHAPEFILE)
                for shapefile in shapefiles:
                    geometries.extend(_get_geometries_from_shapefile(shapefile))
            finally:
                try:
                    shutil.rmtree(tempdir)
                except:
                    pass
        elif is_shapefile(file_name):
            geometries.extend(_get_geometries_from_shapefile(
                file_name, strict=kwargs.get('strict', False)))
        elif os.path.isdir(file_name):
            shp_files = []
            for name in os.listdir(file_name):
                path = os.path.join(file_name, name)
                if not name.startswith('.') and (is_shapefile(name)
                                                 or os.path.isdir(path)):
                    shp_files.append(path)
            geometries.extend(get_geometries(*shp_files))
        else:
            raise RuntimeError('Cannot figure out what to do with file "%s"'
                               % file_name)
    if kwargs.get('get_metadata', None) is False:
        geometries = _strip_metadata(geometries)

    return geometries


def _get_geometries_from_shapefile(shapefile, strict=False):
    """
    NOTE: This is a private method. Use get_geometries() instead.

    Loads the geometry data from a shapefile.

    Returns a list of tuples:

        [ (geometry, {attr: value, attr: value, ...}), ... ]

    The 'attr:value' dictionary are the attributes for each geometry.
    """
    geometries = []
    shp = ogr.Open(shapefile)
    if shp is None:
        raise RuntimeError('Cannot open shapefile "%s".' % shapefile)
    for layer_no in range(shp.GetLayerCount()):
        layer = shp.GetLayer(layer_no)
        layer.ResetReading()
        layer_defn = layer.GetLayerDefn()
        for feature in layer:
            fields = {}
            for field_id in range(layer_defn.GetFieldCount()):
                field_defn = layer_defn.GetFieldDefn(field_id)
                field_name = field_defn.GetName()
                if field_defn.GetType() == ogr.OFTInteger:
                    field_value = feature.GetFieldAsInteger(field_id);
                elif field_defn.GetType() == ogr.OFTReal:
                    field_value = feature.GetFieldAsDouble(field_id);
                elif field_defn.GetType() == ogr.OFTDate \
                        or field_defn.GetType == ogr.OFTDateTime:
                    field_value = feature.GetFieldAsDateTime(field_id)
                else:
                    field_value = feature.GetFieldAsString(field_id);
                if field_name not in fields.keys():
                    fields[field_name] = field_value
                elif (isinstance(fields[field_name], collections.Iterable) and
                      len(fields[field_name]) == 0):
                    fields[field_name] = field_value
            if feature.GetGeometryRef() is None:
                if strict:
                    raise RuntimeError('Feature [%s] has no geometry.'
                                       % feature.GetFID())
                else:
                    print ('Warning! Empty geometry for feature [%s]' % feature.GetFID())
                continue
            geometry = feature.GetGeometryRef().Clone()
            geometries.append((geometry, fields))
    return geometries


def _strip_metadata(geometries):
    """Remove metadata from list of geometries."""
    return [geom for geom, metadata in geometries]


def is_archive(file_name):
    """
    Checks if the file is an archive file.

    Returns the extension of the file.
    """
    file_name = file_name.lower()
    archive_extensions = list(ARCHIVE_EXTENSIONS)

    archive_extensions.sort(key=len, reverse=True)
    for extension in archive_extensions:
        if file_name.endswith('.' + extension):
            return extension
    return False


def extract_all(archive_file_name, location=None):
    """
    Extracts all files from an archive.

    If "location" is specified, the files will be extracted there.
    If "location" is not specified, a temporary directory will be created and
    the files will be extracted in this temporary directory.

    Returns the location where the files were extracted.
    """
    if location is None:
        location = tempfile.mkdtemp()
    if archive_file_name.lower().endswith('zip'):
        zip = zipfile.ZipFile(archive_file_name)
        for name in zip.namelist():
            name = name.replace('\\', '/')
            if name.endswith('/'):
                dir_name = os.path.join(location, name)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
            else:
                zip.extract(name, path=location)
    elif archive_file_name.lower().endswith('tar.bz'):
        tar = tarfile.open(archive_file_name, 'r:bz2')
        tar.extractall(location)
    elif archive_file_name.lower().endswith('tar.gz') or \
            archive_file_name.lower().endswith('tgz'):
        tar = tarfile.open(archive_file_name, 'r:gz')
        tar.extractall(location)
    # TODO: Add support for other archive types here.
    else:
        raise RuntimeError('Could not extract archive "%s"' % archive_file_name)
    return location


def find(directory, regex):
    """
    Searches the directory recursively, looking for file names matching the
    regex.
    Returns the paths to the files matching the regex.
    """
    results = []
    for file_name in sorted(os.listdir(directory)):
        path = os.path.join(directory, file_name)
        if os.path.isdir(path):
            results += find(path, regex)
        else:
            if isinstance(regex, str):
                if file_name == regex:
                    results.append(path)
            else:
                match = regex.match(file_name)
                if match is not None:
                    results.append(path)
    return results
