#include <object_tracker/vector3.h>

namespace TeamKR
{

template <typename _Value>
_Vector3<_Value>::_Vector3()
{
	set(_Value(0), _Value(0), _Value(0));
}

template <typename _Value>
_Vector3<_Value>::_Vector3(_Value _x, _Value _y, _Value _z)
{
	set(_x, _y, _z);
}

template <typename _Value>
_Vector3<_Value>::_Vector3(const _Vector3<_Value>& rhs)
{
	set(rhs.x, rhs.y, rhs.z);
}

template <typename _Value>
_Vector3<_Value>::_Vector3(_Vector3<_Value>& rhs)
{
	set(rhs.x, rhs.y, rhs.z);
}

template <typename _Value>
_Vector3<_Value>& _Vector3<_Value>::operator=(const _Vector3<_Value>& rhs)
{
	set(rhs.x, rhs.y, rhs.z);
	return *this;
}

template <typename _Value>
void _Vector3<_Value>::set(_Value _x, _Value _y, _Value _z)
{
	x = _x;
	y = _y;
	z = _z;
}

template class _Vector3<float>;
template class _Vector3<double>;

}
