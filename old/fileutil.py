#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A class to abstract the implementation details of interacting with
cloud based storage systems like AWS S3

Usage

>>> obj = CloudStorage(cache_path)

S3 File
>>> obj.download('s3://bucket/path/to/file')

Local File
>>> obj.download('file://path/to/file')

Download, cache, and return cache path for file
>>> obj('s3://bucket/path/to/file')


Adding a new Cloud file system

Extend the FileSystem class, implementing each of the methods. The
location of a file on the remote system must be completely specified
by a single string.
    
"""

import shutil
import fsspec
# import boto3
import os        

class GenericFileSystem():
    def __init__(self, root_cache_path, **kwargs):

        #What file systems do we understand
        self.available_filesystems = dict(
            az = AzureFileSystem,
            s3 = S3FileSystem,
            gcs = GoogleCloudFileSystem,
        )
        
        #Classes to interact with invididual remote file systems
        #are cached in this dictionary. By keeping these classes 
        #around, we save on any initialisation costs
        self.active_file_systems = dict()
        
        #Where to cache files downloaded from remote filesystems
        self.root_cache_path = root_cache_path
        self.kwargs = kwargs
        
    def __call__(self, remote):
        return self.cache(remote)
    
    def cache(self, remote):
        """Cache a local copy of file, and return local path of cachefile"""
        
        interface = self.get_protocol(remote)
        cache_path = interface.get_cache_path(self.root_cache_path, remote)
        
        path, file = os.path.split(cache_path)
        os.makedirs(path, exist_ok=True)
    
        self.download(remote, cache_path)
        return cache_path
    
    def read(self, remote):
        """Open a file pointer to a locally cached version of remote file"""
        return open( self.cache(remote, 'r') )
    
    def download(self, remote, local):
        interface = self.get_protocol(remote)
        return interface.download(remote, local)
    
    def upload(self, local, remote):
        interface = self.get_protocol(remote)
        return interface.upload(remote, local)
    
    def exists(self, remote):
        interface = self.get_protocol(remote)
        return interface.exists(remote)
    

    def get_protocol(self, url):
        url_type = url.split(":")[0]
        if len(url_type) == len(url):
            return LocalFileSystem()
        
        try:
            fs = self.available_filesystems[url_type]
        except KeyError:
            raise ValueError("URI type %s not understood" %(url_type))

        if url_type not in self.file_systems:
            self.file_systems[url_type] = fs()
        return self.file_systems[url_type](**self.kwargs)


class AbstractFileSystem():
    """
    Abstract implementation of a cloud file system.
    
    We make a very limited API available
    """
    def __init__(self):
        raise NotImplementedError

    def open(self, remote, *args, **kwargs):
        raise NotImplementedError

    def upload(self, local,remote):
        raise NotImplementedError
        
    def download(self, remote, local):
        raise NotImplementedError    

    def exists(self, remote):
        raise NotImplementedError    
        
    def get_cache_path(self, root, remote):
        raise NotImplementedError    

    def remove(self, remote):
        raise NotImplementedError    


class FsSpecFileSystem(AbstractFileSystem):
    """FsSpec, from the people who made Dask, simplifies the implementation for many filesystems"""

    def __init__(self, identifier, **kwargs):
        self.identifier = 'az'
        self.fs = fsspec.filesystem(self.identifier, **kwargs)

    def open(self, remote, *args, **kwargs):
        """I don't implement close(), so always call this in a context manager
        
        with fs.open() as fp:
            ....
        """
        return self.fs.open(remote, *args, **kwargs)

    def upload(self, local, remote):
        local_fp = open(local)
        with self.fs.open(remote) as remote_fp:
            remote_fp.write(local_fp.read())

    def download(self, remote, local):
        self.fs.download(remote, local)
        # local_fp = open(local, 'w')
        # with self.fs.open(remote) as remote_fp:
        #     local_fp.write(remote_fp.read())

    def exists(self, remote):
        return self.fs.exists(remote)

    def get_cache_path(self, root, remote):
        nchar = len(self.identifier + 3)
        remote = remote[5:]  #Split off az://
        return os.path.join(root, remote)
    
    def remove(self, remote):
        self.fs.rm(remote)


class AzureFileSystem(FsSpecFileSystem):
    def __init__(self, **kwargs):
        FsSpecFileSystem.__init__(self, 'az', **kwargs)
    
class S3FileSystem(FsSpecFileSystem):
    """Untested"""
    def __init__(self, **kwargs):
        FsSpecFileSystem.__init__(self, 's3', **kwargs)

class GoogleCloudFileSystem(FsSpecFileSystem):
    """Untested"""
    def __init__(self, **kwargs):
        FsSpecFileSystem.__init__(self, 'gcs', **kwargs)




class LocalFileSystem(AbstractFileSystem):
    def __init__(self):
        pass
    
    def open(self, remote, *args, **kwargs):
        return open(remote, *args, **kwargs)

    def upload(self, local,remote):
        remote = self.parse_url(remote)
        try:
            return shutil.copy(local, remote)
        except shutil.SameFileError:
            pass
        
    def download(self, remote, local):
        remote = self.parse_url(remote)
        try:
            return shutil.copy(remote, local)
        except shutil.SameFileError:
            pass
        
    def parse_url(self, remote):
        words = remote.split('://')
        
        if len(words) == 1:
            return words[0]
        
        assert words[0] == 'file'
        return words[1]
    
    def exists(self, remote):
        remote = self.parse_url(remote)
        return os.path.exists(remote)

    def get_cache_path(self, root, remote):
        remote = self.parse_url(remote)
        return remote

    def remove(self, remote):
        remote = self.parse_url(remote)
        os.unlink(remote)


try:
    import paramiko
    class SshFileSystem(AbstractFileSystem):
        """Send commands over ssh
        
        URLs are "ssh://user@host.domain.com:/path/to/file"
        
        This is a proof of concept class, and it's very slow. Every
        command has to negotiate opening a new connection. A better implementation
        would be to cache open connections when first used, and re-use them
        for the lifetime of the class. 
        
        TODO
        The url format is "ssh://user@host.domain.com:/path/to/file". It would
        be nice to allow ssh://[user@]host.domain.com:[port:]/path/to/file
        """
        def __init__(self):
            self.client = paramiko.client.SSHClient()
            
            #This is bad. Fix the known_hosts issue correctly
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
        def upload(self, local, remote):
            user, host, path = self.parse_url(remote)
            self.client.connect(host, username=user)
            sftp = self.client.open_sftp()
            sftp.put(local, path)
            sftp.close()
            
        def download(self, remote, local):
            user, host, path = self.parse_url(remote)
            self.client.connect(host, username=user)
            sftp = self.client.open_sftp()
            sftp.get(path, local)
            sftp.close()
            
        def exists(self, remote):
            user, host, path = self.parse_url(remote)
            self.client.connect(host, username=user)
            _, stdout, _ = self.client.exec_command("ls %s" %(path))

            if len(stdout.read()) == 0:
                return False
            return True

        def parse_url(self, remote):
            """TODO Make user optional, have optional port"""
            url = remote[6:]  #strip off 'ssh://'
            user, url = url.split('@')
            host, path = url.split(":")
            
            return user, host, path

        def get_cache_path(self, root_cache_path, remote):
            user, host, path = self.parse_url(remote)
            cachepath = os.path.join(root_cache_path, host, user, path)
            return cachepath 

        def remove(self, remote):
            user, host, path = self.parse_url(remote)
            self.client.connect(host, username=user)
            sftp = self.client.open_sftp()
            sftp.remove(path,)
            sftp.close()
except ImportError:
    pass

#The old interface to s3, kept around just in case    
# class DeprecatedS3FileSystem(FileSystem):
    
#     def __init__(self):
#         #I probably have to do some authentication here
#         self.client = boto3.client('s3')

#     def upload(self, local, remote):
#         bucket, path = self.parse_url(remote)
#         self.client.upload_file(local, bucket, path)
        
#     def download(self, remote, local):
#         bucket, path = self.parse_url(remote)
#         self.client.download_file(bucket, path, local)
        
#     def exists(self, remote):
#         bucket, path = self.parse_url(remote)
#         response = self.client.list_objects_v2(Bucket=bucket, Prefix=path)
        
#         for obj in response.get('Contents', []):
#             if obj['Key'] == path:
#                 return True
#         return False

#     def get_cache_path(self, root, remote):
#         remote = remote[5:]  #Split off s3://
#         return os.path.join(root, remote)
    
#     def remove(self, remote):
#         bucket, path = self.parse_url(remote)
#         self.client.delete_object(Bucket=bucket, Key=path)
        
#     def parse_url(self, remote):
#         #The [5:] cuts out the s3:// at the start
#         words = remote[5:].split('/')
#         bucket = words[0]
#         path = "/".join(words[1:])
#         return bucket, path
    
    


