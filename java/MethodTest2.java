import java.util.Arrays; // java lang 제외 import 필요
import java.util.Random;

public class MethodTest2 {

	public static void main(String[] args) {
		// 난수(Random): 컴퓨터가 불규칙적으로 발생하는 수(greater than or equal to 0.0 and less than 1.0.)
		double a = Math.random();
		System.out.println("Random: " + a);
		// static method는 instance method와 달리 객체 생성 과정이 필요 없음
		// static method 호출 시, 호출하는 class의 method에 속하지 않을 때는 해당 method를 갖고있는 class 명시 필요
		
		Random r  = new Random(); 
		// java가 알고 있는 기본 pkg는 java.lang 밖에 없음
		// 다른 pkg에 해당하는 경우에는 import 필요
		double d = r.nextDouble();
		System.out.println("Random: " + d);
		
//		int ar; // 정수형 변수
		int[] ar = new int[5]; // 배열도 memory 할당 필요 -> 5개의 memory 할당
		// ar[0], ar[1], ar[2], ar[3], ar[4]
		ar[0] = 25;
		ar[1] = 13;
		ar[2] = 45;
		ar[3] = 30;
		ar[4] = 15;
		System.out.println(ar[0] + ", " + ar[1] + ", " + ar[2] + ", " + ar[3] + ", " + ar[4]);

		// 오름차순 정렬
//		System.out.println(Arrays.sort(ar));
		// void: return X -> 변수로 받아줄 수 없음, print로 출력 불가
		Arrays.sort(ar);
		System.out.println(ar[0] + ", " + ar[1] + ", " + ar[2] + ", " + ar[3] + ", " + ar[4]);
		
	}
	
}
