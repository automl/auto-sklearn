for so in `find ~/miniconda/envs/testenv/ -name '*.so'`;
do
    contains_libgcc=`ldd $so | grep libgcc_s.so.1`
        if [ ! -z "$contains_libgcc" ]; then
            echo $so
            path=`ldd $so | grep libgcc_s.so.1`
            echo $path
            path=`echo $path | cut -d " " -f 3`
            file $path
        fi
    done
