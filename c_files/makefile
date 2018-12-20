CC = gcc
FILES = main.c nn.c nn_params.c
OUT_EXE = out
OUT_LIB = nn.so

build: $(FILES)
	$(CC) -Wall -std=c89 -pedantic -lm $(FILES) -o $(OUT_EXE)

lib: $(FILES)
	$(CC) -o $(OUT_LIB) -shared -fPIC $(FILES)

clean:
	rm -f $(OUT_EXE) $(OUT_LIB)
