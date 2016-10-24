This File provides instruction on how to build and config **Virtuoso**, a triple-storage software the package relies on.



##### 1. Download source code from github

```shell
cd tmp
git clone https://github.com/openlink/virtuoso-opensource.git
```



##### 2. Configure and compile the source code to specific path

To build Virtuoso on systems other than `Linux 64-bit`, please refer to the [virtuoso building doc](https://github.com/openlink/virtuoso-opensource)

```shell
cd virtuoso-opensource

# generate makefile
sh autogen.sh 

# PKGPATH is the root directory you put this package in
PKGPATH="put your path here"

# ultimate install path
INSTALLPATH=${PKGPATH}/KnowledgeBase/VirtuosoKG
mkdir -p ${INSTALLPATH}

# flags for Linux 64-bit 
CFLAGS="-O2 -m64"
export CFLAGS

# configurate
./configure --prefix=${INSTALLPATH}

# compile (compiling will take quite a while)
make

# install
make install
```



##### 3. Edit the .ini config file of Virtuoso KB

Here, we config virtuoso in the following way so that a proper performance can be achieved.

```shell
# create a folder to store data to be loaded
cd ${INSTALLPATH}
mkdir data

# edit the .ini config
vi var/lib/virtuoso/db/virtuoso.ini

# all changes necessary to make are under the [Parameters] section

  # 1. DirsAllowed : directory from which data is allowed to be loaded. 
  # So we need to append our created data directory after the default value. 

    # default 
    DirsAllowed = ., ${INSTALLPATH}/share/virtuoso/vad
    # modified
    DirsAllowed = ., ${INSTALLPATH}/share/virtuoso/vad, ${INSTALLPATH}/share/virtuoso/vad

  # 2. MaxQueryMem : maximum memory virtuoso can use to handle queries. 
  # Intuitively, the larger the MaxQueryMem, the potentially faster the query. 
  # The recommemded value is 1/2 to 2/3 of the whole memory on the machine.

    # default
    MaxQueryMem = 2G 
    # modified : for our experiment, on a 6-core machine with 32G memory.
    MaxQueryMem = 16G

  # 3. VectorSize : initial parallel query operations size. 
  # Intuitively, the larger the VectorSize, the potentially faster the query. 
  	
  	# default
    VectorSize = 1000
    # modified : for our experiment, on a 6-core machine with 32G memory.
    VectorSize = 10000
```

