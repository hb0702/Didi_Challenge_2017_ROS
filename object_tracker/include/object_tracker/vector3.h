#pragma once

#include <object_tracker/define.h>

namespace TeamKR
{

template <typename _Value>
class _Vector3
{
public:
	_Vector3();

	_Vector3(_Value _x, _Value _y, _Value _z);

	_Vector3(const _Vector3& rhs);

	_Vector3(_Vector3& rhs);

	_Vector3& operator=(const _Vector3& rhs);

	void set(_Value _x, _Value _y, _Value _z);

public:
	_Value x;
	_Value y;
	_Value z;
};

typedef _Vector3<value_type> Vector3;

}
