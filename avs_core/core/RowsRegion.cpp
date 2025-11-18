#include <avs/cpuid.h>

static int EnvCheckForRRSize()
{
    int iRRSize = 32000; // set init value to about 32 kByte (tests shows best about L1D cache size ?)

    // Todo: calculate recommended auto value as about CPU_cache_size/(Threads_num * 3) - need test for best performance value

    return iRRSize;
}


class _RowsRegion
{
private:

    int RowsRegionSize;
    _RowsRegion() { RowsRegionSize = EnvCheckForRRSize(); }

public:
    static _RowsRegion& getInstance() {
        static _RowsRegion theInstance;
        return theInstance;
    }

    int GetRowsRegionSize() {
        return RowsRegionSize;
    }

    void SetRowsRegionSize(int new_size) {
        RowsRegionSize = new_size;
    }
};

int GetRowsRegionSize() {
    return _RowsRegion::getInstance().GetRowsRegionSize();
}

void SetRowsRegionSize(int new_size) {
    _RowsRegion::getInstance().SetRowsRegionSize(new_size);
}
