#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL BINTOOLS_ARRAY_API
#include <numpy/arrayobject.h>

#include <stdbool.h>
#include <stddef.h>

#define error_converting(x)  (((x) == -1) && PyErr_Occurred())

/* Find the minimum and maximum of an integer array */
static void
minmax(const npy_intp *data, npy_intp data_len, npy_intp *mn, npy_intp *mx)
{
    npy_intp min = *data;
    npy_intp max = *data;

    while (--data_len) {
        const npy_intp val = *(++data);
        if (val < min) {
            min = val;
        }
        else if (val > max) {
            max = val;
        }
    }

    *mn = min;
    *mx = max;
}

NPY_NO_EXPORT PyObject *
arr_binreduce(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *op = NULL, *list = NULL, *weight = Py_None, *mlength = NULL;
    PyObject *bop = NULL;
    PyArrayObject *lst = NULL, *ans = NULL, *wts = NULL;
    npy_intp *numbers, *ians, len, mx, mn, ans_size;
    npy_intp minlength = 0;
    npy_intp i;
    double *weights , *dans;
    char *cop;
    static char *kwlist[] = {"op", "list", "weights", "minlength", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!OO|O:binreduce",
                kwlist, &PyUnicode_Type, &op, &list, &weight, &mlength)) {
            goto fail;
    }

    bop = PyUnicode_AsASCIIString(op);
    if (bop == NULL) {
        goto fail;
    }
    cop = PyBytes_AS_STRING(bop);

    lst = (PyArrayObject *)PyArray_ContiguousFromAny(list, NPY_INTP, 1, 1);
    if (lst == NULL) {
        goto fail;
    }
    len = PyArray_SIZE(lst);

    /*
     * This if/else if can be removed by changing the argspec to O|On above,
     * once we retire the deprecation
     */
    if (mlength == Py_None) {
        /* NumPy 1.14, 2017-06-01 */
        if (DEPRECATE("0 should be passed as minlength instead of None; "
                      "this will error in future.") < 0) {
            goto fail;
        }
    }
    else if (mlength != NULL) {
        minlength = PyArray_PyIntAsIntp(mlength);
        if (error_converting(minlength)) {
            goto fail;
        }
    }

    if (minlength < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "'minlength' must not be negative");
        goto fail;
    }

    /* handle empty list */
    if (len == 0) {
        ans = (PyArrayObject *)PyArray_ZEROS(1, &minlength, NPY_INTP, 0);
        if (ans == NULL){
            goto fail;
        }
        Py_DECREF(lst);
        return (PyObject *)ans;
    }

    numbers = (npy_intp *)PyArray_DATA(lst);
    minmax(numbers, len, &mn, &mx);
    if (mn < 0) {
        PyErr_SetString(PyExc_ValueError,
                "'list' argument must have no negative elements");
        goto fail;
    }
    ans_size = mx + 1;
    if (mlength != Py_None) {
        if (ans_size < minlength) {
            ans_size = minlength;
        }
    }
    wts = (PyArrayObject *)PyArray_ContiguousFromAny(
                                            weight, NPY_DOUBLE, 1, 1);
    if (wts == NULL) {
        goto fail;
    }
    weights = (double *)PyArray_DATA(wts);
    if (PyArray_SIZE(wts) != len) {
        PyErr_SetString(PyExc_ValueError,
                "The weights and list don't have the same length.");
        goto fail;
    }
    ans = (PyArrayObject *)PyArray_ZEROS(1, &ans_size, NPY_DOUBLE, 0);
    if (ans == NULL) {
        goto fail;
    }
    dans = (double *)PyArray_DATA(ans);
    for (i = 0; i < ans_size; i++)
        dans[i] = NAN;
    if (!strcmp(cop, "max")) {
        NPY_BEGIN_ALLOW_THREADS;
        for (i = 0; i < len; i++) {
            if (dans[numbers[i]] == NAN) {
                dans[numbers[i]] = weights[i];
            }
            else {
                dans[numbers[i]] = fmax(dans[numbers[i]], weights[i]);          
            }
        }
        NPY_END_ALLOW_THREADS;
    }
    else if (!strcmp(cop, "min")) {
        NPY_BEGIN_ALLOW_THREADS;
        for (i = 0; i < len; i++) {
            if (dans[numbers[i]] == NAN) {
                dans[numbers[i]] = weights[i];
            }
            else {
                dans[numbers[i]] = fmin(dans[numbers[i]], weights[i]);          
            }
        }
        NPY_END_ALLOW_THREADS;
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                        "'op' unknown.");
        goto fail;
    }
    Py_DECREF(bop);
    Py_DECREF(lst);
    Py_DECREF(wts);
    return (PyObject *)ans;

fail:
    Py_XDECREF(bop);
    Py_XDECREF(lst);
    Py_XDECREF(wts);
    Py_XDECREF(ans);
    return NULL;
}

/*
 * Method declaration
 */

static PyMethodDef module_methods[] = {
    {"binreduce", (PyCFunction)arr_binreduce, METH_VARARGS | METH_KEYWORDS,
     NULL},
    { NULL, NULL, 0, NULL }  /* Sentinel */
};

/*
 * Module initialization
 */

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

/*
 * Module declaration
 */

#if PY_MAJOR_VERSION >= 3
    #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
    #define MOD_DEF(ob, name, methods, doc) \
        static struct PyModuleDef moduledef = { \
            PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
        ob = PyModule_Create(&moduledef);
#else
    #define MOD_INIT(name) PyMODINIT_FUNC init##name(void)
    #define MOD_DEF(ob, name, methods, doc) \
        ob = Py_InitModule3(name, methods, doc);
#endif

MOD_INIT(_bintools)
{
    PyObject* m;

    import_array();

    MOD_DEF(m, "_bintools", module_methods,
            "C support functions for bintools.");

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
