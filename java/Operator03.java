public class Operator03 {

	public static void main(String[] args) {		
		int a = 5;
		a += 2; // 7
		a *= 3; // 21 a = a*2; 2항 연산자 사용 시, 연산자와 =를 붙여서 사용해야 함
		a /= 5; // 4

		System.out.println("a = " + a);
	
		a++;
		System.out.println("a = " + a); // 5
		
		int b = a++; // int b = a -> a = a+1
		System.out.println("a = " + a + ", b = " + b);

		int c = ++a * b--; // ++a -> a*b -> b--
		System.out.println("a = " + a + ", b = " + b + ", c = " + c);

		System.out.println("a++ = " + a++);
		System.out.println("a = " + a);
		
	}
	
}


/*
3항 연산자: a = a + 2;
2항 연산자: a + =2;
1항(단항) 연산자: a++ (초기값 유지) / ++a (초기값 증가)
 */
