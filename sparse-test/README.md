If you want to use the Justfile:

To emit the LLVM IR, do `just llvm-emit <filename>` but leave the file extension out:
e.g.

```
> ls
test.c

> just llvm-emit test
```


To compile `C` program, do `just compile <filename>` but leave the file extension out:
e.g.

```
> ls
test.c

> just compile test

> ./test
```
