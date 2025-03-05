<!-- markdownlint-disable -->

# <kbd>module</kbd> `animate`





---

## <kbd>function</kbd> `animate_trajectory`

```python
animate_trajectory(
    positions,
    quaternions,
    box_dimensions=(0.5, 1.0, 0.5),
    total_time=10.0,
    save_animation=False,
    save_path=None
)
```

Animate a 3D trajectory with a box moving along it, using quaternions for orientation. 



**Parameters:**
 
----------- positions : array-like  Array of shape (n, 3) containing the x, y, z positions along the trajectory. quaternions : array-like  Array of shape (n, 4) containing the orientation quaternions [w, x, y, z]. box_dimensions : tuple, optional  The dimensions of the box (length, width, height), defaults to (0.5, 1.0, 0.5). total_time : float, optional  The total time of the animation in seconds, defaults to 10.0. save_animation : bool, optional  Whether to save the animation as a file, defaults to False. save_path : str, optional  Path to save the animation file. Required if save_animation is True. 



**Returns:**
 
-------- anim : FuncAnimation  The animation object. 


---

## <kbd>function</kbd> `generate_sample_trajectory`

```python
generate_sample_trajectory()
```

Generate a sample helical trajectory with quaternion orientations. 


---

## <kbd>function</kbd> `rotation_matrix_to_quaternion`

```python
rotation_matrix_to_quaternion(R)
```

Convert a 3x3 rotation matrix to a quaternion [w, x, y, z]. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
